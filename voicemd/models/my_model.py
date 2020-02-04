import logging
import torch.nn as nn

from voicemd.utils.hp_utils import check_and_log_hp

logger = logging.getLogger(__name__)


class MyModel(nn.Module):

    def __init__(self, hyper_params):
        super(MyModel, self).__init__()

        check_and_log_hp(['size'], hyper_params)
        self.hyper_params = hyper_params
        self.linear1 = nn.Linear(5, hyper_params['size'])
        self.linear2 = nn.Linear(hyper_params['size'], 1)

    def forward(self, data):
        hidden = self.linear1(data)
        result = self.linear2(hidden)
        return result.squeeze()
