import logging
import os
import wandb
import mlflow
import orion
import yaml
import time
import torch
import tqdm
import numpy as np
import pytz

from sklearn.metrics import confusion_matrix
from mlflow import log_metric
from orion.client import report_results
from yaml import dump
from yaml import load
from datetime import datetime

logger = logging.getLogger(__name__)

BEST_MODEL_NAME = 'best_model.pt'
LAST_MODEL_NAME = 'last_model.pt'
STAT_FILE_NAME = 'stats.yaml'


def reload_model(output, model_name, model, optimizer, start_from_scratch=False):
    saved_model = os.path.join(output, model_name)
    if start_from_scratch and os.path.exists(saved_model):
        logger.info('saved model file "{}" already exists - but NOT loading it '
                    '(cause --start_from_scratch)'.format(output))
        return
    if os.path.exists(saved_model):
        logger.info('saved model file "{}" already exists - loading it'.format(output))
        model.load_state_dict(torch.load(saved_model))

        stats = load_stats(output)
        logger.info('model status: {}'.format(stats))
        return stats
    if os.path.exists(output):
        logger.info('saved model file not found - but output folder exists already - keeping it')
        return

    logger.info('no saved model file found - nor output folder - creating it')
    os.makedirs(output)


def write_stats(output, best_eval_score, epoch, remaining_patience):
    to_store = {'best_dev_metric': best_eval_score, 'epoch': epoch,
                'remaining_patience': remaining_patience,
                'mlflow_run_id': mlflow.active_run().info.run_id}
    with open(os.path.join(output, STAT_FILE_NAME), 'w') as stream:
        dump(to_store, stream)


def load_stats(output):
    with open(os.path.join(output, STAT_FILE_NAME), 'r') as stream:
        stats = load(stream, Loader=yaml.FullLoader)
    return stats['best_dev_metric'], stats['epoch'], stats['remaining_patience'], \
        stats['mlflow_run_id']


def train(model, optimizer, scheduler, loss_fun, train_loader, dev_loader, patience, output,
          max_epoch, use_progress_bar=True, start_from_scratch=False):

    try:
        best_dev_metric = train_impl(
            dev_loader, loss_fun, max_epoch, model, optimizer, scheduler, output,
            patience, train_loader, use_progress_bar, start_from_scratch)
    except RuntimeError as err:
        if orion.client.IS_ORION_ON and 'CUDA out of memory' in str(err):
            logger.error(err)
            logger.error('model was out of memory - assigning a bad score to tell Orion to avoid'
                         'too big model')
            best_dev_metric = -999
        else:
            raise err

    report_results([dict(
        name='dev_metric',
        type='objective',
        # note the minus - cause orion is always trying to minimize (cit. from the guide)
        value=-float(best_dev_metric))])


tz_NY = pytz.timezone('America/New_York')
datetime_NY = datetime.now(tz_NY)


def train_impl(dev_loader, loss_fun, max_epoch, model, optimizer, scheduler, output, patience,
               train_loader, use_progress_bar, start_from_scratch=False):

    timestamp = datetime.now(pytz.timezone('America/New_York'))
    wandb.init(project="voicemd", name=timestamp.strftime('%H:%M:%S'))

    if use_progress_bar:
        pb = tqdm.tqdm
    else:
        def pb(x, total):
            return x

    stats = reload_model(output, LAST_MODEL_NAME, model, optimizer, start_from_scratch)
    if stats is None:
        best_dev_metric = None
        remaining_patience = patience
        start_epoch = 0
    else:
        best_dev_metric, start_epoch, remaining_patience, _ = stats

    if remaining_patience <= 0:
        logger.warning(
            'remaining patience is zero - not training (and returning best dev score {})'.format(
                best_dev_metric))
        return best_dev_metric
    if start_epoch >= max_epoch:
        logger.warning(
            'start epoch {} > max epoch {} - not training (and returning best dev score '
            '{})'.format(start_epoch, max_epoch, best_dev_metric))
        return best_dev_metric
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # TODO : Move this somewhere more useful
    def get_performance_metrics(outputs, model_target):
        probs = torch.softmax(outputs, 1).detach().numpy() > 0.5
        preds = np.argmax(probs, 1)
        targs = model_target.detach().numpy()
        acc = np.sum(np.equal(preds, targs)) / len(preds)
        conf_mat = confusion_matrix(targs, preds)
        return acc, conf_mat

    val_acc = []
    flag = 0

    for epoch in range(start_epoch, max_epoch):

        start = time.time()
        # train
        train_cumulative_loss = 0.0
        train_cumulative_acc = 0.0
        train_cumulative_conf_mat = np.zeros((2, 2))
        examples = 0
        model.train()
        train_steps = len(train_loader)

        for i, data in pb(enumerate(train_loader, 0), total=train_steps):

            model_input, model_target = data
            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(model_input.to(device))
            model_target = model_target.type(torch.long)
            model_target = model_target.to(device)
            loss = loss_fun(outputs, model_target)
            loss.backward()
            optimizer.step()

            acc, conf_mat = get_performance_metrics(outputs, model_target)

            train_cumulative_loss += loss.item()
            train_cumulative_acc += acc
            train_cumulative_conf_mat += conf_mat
            examples += model_target.shape[0]

        train_end = time.time()
        avg_train_loss = train_cumulative_loss / examples
        avg_train_acc = train_cumulative_acc / train_steps
        print(train_cumulative_conf_mat)

        # dev
        model.eval()
        dev_steps = len(dev_loader)
        dev_cumulative_loss = 0.0
        dev_cumulative_acc = 0.0
        dev_cumulative_conf_mat = np.zeros((2, 2))
        examples = 0
        total_loss, n_correct, n_samples = 0.0, 0, 0

        for i, data in pb(enumerate(dev_loader, 0), total=dev_steps):
            model_input, model_target = data
            with torch.no_grad():

                # model_input = model_input.to(device),
                # model_target.to(device)
                # pred_label = model(model_input)

                # loss = criterion(pred_label, model_target)
                #
                # _, y_label_ = torch.max(model_target, 1)
                # n_correct += (y_label_ == model_target).sum().item()
                # total_loss += loss.item() * model_input.shape[0]
                # n_samples += model_input.shape[0]
                #
                # val_loss = total_loss / n_samples
                # val_acc = n_correct / n_samples * 100

                outputs = model(model_input.to(device))
                model_target = model_target.type(torch.long)
                model_target = model_target.to(device)
                loss = loss_fun(outputs, model_target)
                dev_cumulative_loss += loss.item()

                acc, conf_mat = get_performance_metrics(outputs, model_target)
                dev_cumulative_acc += acc
                dev_cumulative_conf_mat += conf_mat

            examples += model_target.shape[0]

        print(dev_cumulative_conf_mat)
        avg_dev_loss = dev_cumulative_loss / examples
        avg_dev_acc = dev_cumulative_acc / dev_steps
        log_metric("train_loss", avg_train_loss, step=epoch)
        log_metric("train_acc", avg_train_acc, step=epoch)
        log_metric("dev_loss", avg_dev_loss, step=epoch)
        log_metric("dev_acc", avg_dev_acc, step=epoch)

        dev_end = time.time()
        torch.save(model.state_dict(), os.path.join(output, LAST_MODEL_NAME))

        if best_dev_metric is None or avg_dev_acc > best_dev_metric:
            best_dev_metric = avg_dev_acc
            remaining_patience = patience
            torch.save(model.state_dict(), os.path.join(output, BEST_MODEL_NAME))
        else:
            remaining_patience -= 1

        logger.info(
            'done #epoch {:3} => loss {:5.3f}, acc {:5.3f}- dev loss {:3.4f} dev-acc {:5.3f}, (will try for {} more epoch) - '
            'train min. {:4.2f} / dev min. {:4.2f}'.format(
                epoch, avg_train_loss,
                avg_train_acc,
                avg_dev_loss,
                avg_dev_acc,
                remaining_patience,
                (train_end - start) / 60,
                (dev_end - train_end) / 60)
        )

        # get current lr
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']

        val_acc.append(avg_dev_acc)

        scheduler.step()

        # if epoch > 100:
        #
        #     if val_acc[-1] < max(val_acc[:-1]):
        #         flag += 1
        #
        #     if flag == 15:
        #         for param_group in optimizer.param_groups:
        #             new_lr = current_lr * 0.5
        #             param_group['lr'] = new_lr
        #             flag = 0

        wandb.log({"Val Accuracy": avg_dev_acc, "Val Loss": avg_dev_loss, "Train Loss": avg_train_loss,
                   "Train Accuracy": avg_train_acc, "Learning Rate": current_lr, "Flag Count": flag})

        write_stats(output, best_dev_metric, epoch + 1, remaining_patience)
        log_metric("best_dev_metric", best_dev_metric)

        if remaining_patience <= 0:
            logger.info('done! best dev metric is {}'.format(best_dev_metric))
            break
    logger.info('training completed (epoch done {} - max epoch {})'.format(epoch + 1, max_epoch))
    logger.info('Finished Training')
    return best_dev_metric


#python voicemd/main.py --data /Users/alex/github/voicemd/data/voice_clips/ --output output --config voicemd/config.yaml  --start_from_scratch
