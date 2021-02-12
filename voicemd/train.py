import json
import logging
import os
import mlflow
import orion
import yaml
import pickle
import time
import torch
import tqdm
import numpy as np

from sklearn.metrics import confusion_matrix
from mlflow import log_metric
from orion.client import report_results
from yaml import dump
from yaml import load

from voicemd.eval import evaluate_loaders, get_batch_performance_metrics, predictions_from_probs

logger = logging.getLogger(__name__)

BEST_MODEL_NAME = "best_model"
LAST_MODEL_NAME = "last_model"
STAT_FILE_NAME = "stats.yaml"


def reload_model(output_dir, model_name, model, optimizer, start_from_scratch=False):
    saved_model = os.path.join(output_dir, model_name)
    if start_from_scratch and os.path.exists(saved_model):
        logger.info(
            'saved model file "{}" already exists - but NOT loading it '
            "(cause --start_from_scratch)".format(output_dir)
        )
        return
    if os.path.exists(saved_model):
        logger.info('saved model file "{}" already exists - loading it'.format(output_dir))
        model.load_state_dict(torch.load(saved_model))

        stats = load_stats(output_dir)
        logger.info("model status: {}".format(stats))
        return stats
    if os.path.exists(output_dir):
        logger.info(
            "saved model file not found - but output_dir folder exists already - keeping it"
        )
        return

    logger.info("no saved model file found - nor output_dir folder - creating it")
    os.makedirs(output_dir)


def write_stats(output_dir, best_eval_score, epoch, remaining_patience):
    to_store = {
        "best_dev_metric": best_eval_score,
        "epoch": epoch,
        "remaining_patience": remaining_patience,
        "mlflow_run_id": mlflow.active_run().info.run_id,
    }
    with open(os.path.join(output_dir, STAT_FILE_NAME), "w") as stream:
        dump(to_store, stream)


def load_stats(output_dir):
    with open(os.path.join(output_dir, STAT_FILE_NAME), "r") as stream:
        stats = load(stream, Loader=yaml.FullLoader)
    return (
        stats["best_dev_metric"],
        stats["epoch"],
        stats["remaining_patience"],
        stats["mlflow_run_id"],
    )


def train(
    hyper_params,
    model,
    optimizer,
    loss_fun,
    train_loader,
    valid_loaders,
    test_loaders,
    patience,
    output_dir,
    max_epoch,
    split_number,
    use_progress_bar=True,
    start_from_scratch=False,
):

    try:
        best_dev_metric = train_impl(
            hyper_params,
            train_loader,
            valid_loaders,
            test_loaders,
            loss_fun,
            max_epoch,
            model,
            optimizer,
            output_dir,
            patience,
            split_number,
            use_progress_bar,
            start_from_scratch,
        )
    except RuntimeError as err:
        if orion.client.IS_ORION_ON and "CUDA out of memory" in str(err):
            logger.error(err)
            logger.error(
                "model was out of memory - assigning a bad score to tell Orion to avoid"
                "too big model"
            )
            best_dev_metric = -999
        else:
            raise err

    #  For orion
    #  report_results([dict(
    #      name='dev_metric',
    #      type='objective',
    #      # note the minus - cause orion is always trying to minimize (cit. from the guide)
    #      value=-float(best_dev_metric))])


def train_valid_step(hyper_params, loader, model, optimizer, loss_fun, device, split):
    n_ages = len(hyper_params['age_label2cat'])
    n_genders = len(hyper_params['gender_label2cat'])
    stats = {
        'total_loss': 0,
        'gender_loss': 0,
        'gender_acc': 0,
        'gender_conf_mat': np.zeros((n_genders, n_genders)),
        'age_loss': 0,
        'age_acc': 0,
        'age_conf_mat': np.zeros((n_ages, n_ages)),
        'sample_count': 0,
        'step_count': 0,
    }
    # prob_stats only relevant in eval mode.
    # When in eval mode, we collect all probabilities since each loader represents a patient
    assert split in ['train', 'eval']
    if split == 'train':
        model.train()
    elif split == 'eval':
        model.eval()
        stats['gender_probs'] = []
        stats['age_probs'] = []

    for data in loader:
        model_input, model_targets = data
        # forward + backward + optimize
        if split == 'train':
            optimizer.zero_grad()
        outputs = model(model_input.to(device))
        batch_loss = 0

        categories = hyper_params['categories']

        for cat in categories:
            output = outputs[cat]
            model_target = model_targets[cat]
            loss_fn = loss_fun[cat]

            model_target = model_target.type(torch.long).to(device)
            cat_loss = loss_fn(output, model_target)
            batch_loss += cat_loss
            stats[f'{cat}_loss'] += cat_loss.item()
            stats['total_loss'] += cat_loss.item()

            if split == 'eval':
                # collect probas computed on each frame
                cat_prob_frame = torch.nn.functional.softmax(outputs[cat], dim=1)
                stats[f'{cat}_probs'].extend(cat_prob_frame.detach().numpy())


            cat_acc, cat_conf_mat = get_batch_performance_metrics(
                outputs[cat],
                model_targets[cat],
                labels=list(hyper_params[f'{cat}_label2cat'].values())
            )
            stats[f'{cat}_conf_mat'] += cat_conf_mat
            stats[f'{cat}_acc'] += cat_acc

        stats['sample_count'] += model_targets['gender'].shape[0]
        stats['step_count'] += 1

        if split == 'train':
            batch_loss.backward()
            optimizer.step()

    stats[f'total_avg_loss'] = stats['total_loss'] / stats['sample_count']
    for cat in categories:
        stats[f'{cat}_avg_acc'] = stats[f'{cat}_acc'] / stats['step_count']
        stats[f'{cat}_avg_loss'] = stats[f'{cat}_loss'] / stats['sample_count']

    if split == 'eval':
        # compute predictions per patient, return results

        stats['uid'] = loader.dataset.metadata.index[0]
        for cat in categories:
            stats[f'{cat}_prediction'], stats[f'{cat}_confidence'] = predictions_from_probs(stats[f'{cat}_probs'])
            stats[f'{cat}_gt'] = int(model_targets[cat][0]) # in eval, they all come from same sample

    return stats


def train_impl(
    hyper_params,
    train_loader,
    valid_loaders,
    test_loaders,
    loss_fun,
    max_epoch,
    model,
    optimizer,
    output_dir,
    patience,
    split_number,
    use_progress_bar,
    start_from_scratch=False,
    use_scheduler=True,
):

    if use_progress_bar:
        pb = tqdm.tqdm
    else:

        def pb(x, total):
            return x

    stats = reload_model(output_dir, LAST_MODEL_NAME, model, optimizer, start_from_scratch)
    if stats is None:
        best_dev_metric = None
        remaining_patience = patience
        start_epoch = 0
    else:
        best_dev_metric, start_epoch, remaining_patience, _ = stats

    if remaining_patience <= 0:
        logger.warning(
            "remaining patience is zero - not training (and returning best dev score {})".format(
                best_dev_metric
            )
        )
        return best_dev_metric
    if start_epoch >= max_epoch:
        logger.warning(
            "start epoch {} > max epoch {} - not training (and returning best dev score "
            "{})".format(start_epoch, max_epoch, best_dev_metric)
        )
        return best_dev_metric
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    for epoch in range(start_epoch, max_epoch):

        start = time.time()

        train_stats = train_valid_step(hyper_params, train_loader, model, optimizer, loss_fun, device, split='train')
        avg_train_loss = train_stats['total_loss'] / train_stats['sample_count']
        avg_train_acc = train_stats['gender_acc'] / train_stats['step_count']

        train_end = time.time()
        for cat in hyper_params['categories']:
            logger.info(
                "Confidence matrix for train: \n {}".format(train_stats[f'{cat}_conf_mat'])
            )
        log_metric("train_loss", avg_train_loss, step=epoch)
        log_metric("train_acc", avg_train_acc, step=epoch)

        # Validation
        model.eval()
        validation_results = evaluate_loaders(
            hyper_params, valid_loaders, model, loss_fun, device, pb
        )

        if use_scheduler:
            scheduler.step()

        avg_eval_acc = 0
        avg_eval_loss = 0
        for cat in hyper_params['categories']:
            logger.info("results for {}: ".format(cat))
            logger.info(
                "Confidence matrix on every validation spectrum: \n {}".format(
                    validation_results[f"{cat}_conf_mat_spectrums"]
                )
            )
            logger.info(
                "Confidence matrix per validation patient: \n {}".format(
                    validation_results[f"{cat}_conf_mat_patients"]
                )
            )

            log_metric(f"{cat}_eval_loss", validation_results[f"{cat}_avg_loss"], step=epoch)
            log_metric(f"{cat}_eval_acc", validation_results[f"{cat}_avg_acc"], step=epoch)

            avg_eval_acc += validation_results[f"{cat}_avg_acc"]
            avg_eval_loss += validation_results[f"{cat}_avg_loss"]
        avg_eval_acc = avg_eval_acc / len(hyper_params['categories'])
        avg_eval_loss = avg_eval_loss / len(hyper_params['categories'])
        torch.save(model.state_dict(), os.path.join(output_dir, LAST_MODEL_NAME))
        if best_dev_metric is None or avg_eval_acc > best_dev_metric:
            best_dev_metric = avg_eval_acc
            remaining_patience = patience
            best_model_split_name = BEST_MODEL_NAME + '_split_' + str(split_number)
            torch.save(model.state_dict(), os.path.join(output_dir, best_model_split_name))
            #  logger.info()
            logger.info("New best model, saving results.\n")
        else:
            remaining_patience -= 1

        logger.info(
            "done #epoch {:3} => loss {:5.3f}, acc {:5.3f}- dev loss {:3.4f} dev-acc {:5.3f}, (will try for {} more epoch) - ".format(
                epoch,
                avg_train_loss,
                avg_train_acc,
                avg_eval_loss,
                avg_eval_acc,
                remaining_patience,
            )
        )

        write_stats(output_dir, best_dev_metric, epoch + 1, remaining_patience)
        log_metric("best_dev_metric", best_dev_metric)

        if remaining_patience <= 0:
            logger.info("done! best dev metric is {}".format(best_dev_metric))
            break
    logger.info(
        "training completed (epoch done {} - max epoch {})".format(epoch + 1, max_epoch)
    )
    logger.info("Finished Training")

    # Evaluate on test set
    logger.info("Evaluating on test set:")
    model.load_state_dict(torch.load(output_dir + '/' + best_model_split_name))  # load the best model
    model.eval()
    test_results = evaluate_loaders(hyper_params, test_loaders, model, loss_fun, device, pb)
    for cat in hyper_params['categories']:
        logger.info("results for {}: ".format(cat))
        logger.info(
            "Confidence matrix on every test spectrum: \n {}".format(
                test_results[f"{cat}_conf_mat_spectrums"]
            )
        )
        logger.info(
            "Confidence matrix per test patient: \n {}".format(
                test_results[f"{cat}_conf_mat_patients"]
            )
        )
    logger.info("saving results.")
    with open(output_dir + '/test_results_split_' + str(split_number) + '.pkl', 'wb') as out:
        pickle.dump(test_results, out)

    return best_dev_metric
