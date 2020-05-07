import logging
import torch
from torch import optim

from voicemd.models.my_model import MyModel
from voicemd.models.densenet import densenet121, densenet_small
from voicemd.models.simple_cnn import SimpleCNN
from voicemd.models.long_filter_cnn import LongFilterCNN

logger = logging.getLogger(__name__)


def load_model(hyper_params):
    architecture = hyper_params['architecture']
    # __TODO__ fix architecture list
    if architecture == 'my_model':
        model_class = MyModel

    elif architecture == 'densenet121':
        model_class = densenet121

    elif architecture == 'densenet_small':
        model_class = densenet_small

    elif architecture == 'simplecnn':
        model_class = SimpleCNN

    elif architecture == 'longfilter':
        model_class = LongFilterCNN

    else:
        raise ValueError('architecture {} not supported'.format(architecture))
    logger.info('selected architecture: {}'.format(architecture))

    model = model_class(hyper_params)
    logger.info(model)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info('using device {}'.format(device))
    if torch.cuda.is_available():
        logger.info(torch.cuda.get_device_name(0))

    return model


def load_optimizer(hyper_params, model):
    optimizer_name = hyper_params['optimizer']
    lr = hyper_params['learning_rate']
    # __TODO__ fix optimizer list
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError('optimizer {} not supported'.format(optimizer_name))
    return optimizer


def load_loss(hyper_params, train_loader=None):

    # Use the proportion from train_loader to weigh the loss since it can be unbalanced classes
    if train_loader:
        n_male = sum(train_loader.dataset.metadata['gender'] == 'M')
        n_female = sum(train_loader.dataset.metadata['gender'] == 'F')
        n_total = n_male + n_female

        # Male is label 1, female is label 0, use the proportion of the other to weigh the loss
        weight = torch.tensor([n_male/n_total, n_female/n_total])

    else:
        weight = None

    return torch.nn.CrossEntropyLoss(weight=weight)
