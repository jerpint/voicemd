import logging
import os

import mlflow
import orion
import yaml
import time
import torch
import tqdm
from mlflow import log_metric
from orion.client import report_results
from yaml import dump
from yaml import load

logger = logging.getLogger(__name__)

BEST_MODEL_NAME = 'best_model'
LAST_MODEL_NAME = 'last_model'
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


def train(model, optimizer, loss_fun, train_loader, dev_loader, patience, output,
          max_epoch, use_progress_bar=True, start_from_scratch=False):

    try:
        best_dev_metric = train_impl(
            dev_loader, loss_fun, max_epoch, model, optimizer, output,
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


def train_impl(dev_loader, loss_fun, max_epoch, model, optimizer, output, patience,
               train_loader, use_progress_bar, start_from_scratch=False):

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

    for epoch in range(start_epoch, max_epoch):

        start = time.time()
        # train
        train_cumulative_loss = 0.0
        examples = 0
        model.train()
        train_steps = len(train_loader)
        for i, data in pb(enumerate(train_loader, 0), total=train_steps):
            model_input, model_target = data
            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(model_input.to(device))
            model_target = torch.squeeze(model_target.to(device))
            loss = loss_fun(outputs, model_target)
            loss.backward()
            optimizer.step()

            train_cumulative_loss += loss.item()
            examples += model_target.shape[0]

        train_end = time.time()
        avg_train_loss = train_cumulative_loss / examples

        # dev
        model.eval()
        dev_steps = len(dev_loader)
        dev_cumulative_loss = 0.0
        examples = 0
        for i, data in pb(enumerate(dev_loader, 0), total=dev_steps):
            model_input, model_target = data
            with torch.no_grad():
                outputs = model(model_input.to(device))
                model_target = torch.squeeze(model_target.to(device))
                loss = loss_fun(outputs, model_target)
                dev_cumulative_loss += loss.item()
            examples += model_target.shape[0]

        avg_dev_loss = dev_cumulative_loss / examples
        log_metric("dev_loss", avg_dev_loss, step=epoch)
        log_metric("train_loss", avg_train_loss, step=epoch)

        dev_end = time.time()
        torch.save(model.state_dict(), os.path.join(output, LAST_MODEL_NAME))

        if best_dev_metric is None or avg_dev_loss < best_dev_metric:
            best_dev_metric = avg_dev_loss
            remaining_patience = patience
            torch.save(model.state_dict(), os.path.join(output, BEST_MODEL_NAME))
        else:
            remaining_patience -= 1

        logger.info(
            'done #epoch {:3} => loss {:5.3f} - dev loss {:3.2f} (will try for {} more epoch) - '
            'train min. {:4.2f} / dev min. {:4.2f}'.format(
                epoch, avg_train_loss, avg_dev_loss, remaining_patience, (train_end - start) / 60,
                (dev_end - train_end) / 60))

        write_stats(output, best_dev_metric, epoch + 1, remaining_patience)
        log_metric("best_dev_metric", best_dev_metric)

        if remaining_patience <= 0:
            logger.info('done! best dev metric is {}'.format(best_dev_metric))
            break
    logger.info('training completed (epoch done {} - max epoch {})'.format(epoch + 1, max_epoch))
    logger.info('Finished Training')
    return best_dev_metric
