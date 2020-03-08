import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

#  from voicemd.utils.hp_utils import check_and_log_hp

logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):

    def __init__(self, hyper_params):
        super(SimpleCNN, self).__init__()
        self.hyper_params = hyper_params
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 64, 3, 1),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            )

        self.classifier = nn.Sequential(
            nn.Linear(12992, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):

        x = self.convs(x)
        x = torch.flatten(x, 1)
        output = self.classifier(x)

        return output
