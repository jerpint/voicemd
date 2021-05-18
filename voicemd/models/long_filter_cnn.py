import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

#  from voicemd.utils.hp_utils import check_and_log_hp
from voicemd.eval import get_num_categories

logger = logging.getLogger(__name__)


class LongFilterCNN(nn.Module):

    def __init__(self, hyper_params):
        super(LongFilterCNN, self).__init__()
        self.hyper_params = hyper_params
        self.n_ages = get_num_categories(hyper_params['age_label2cat'])
        self.n_genders = get_num_categories(self.hyper_params['gender_label2cat'])
        self.conv2d = nn.Sequential(
            nn.Conv2d(1, 64, (80, 3), 1),
            nn.ReLU(),
        )

        self.conv1d = nn.Sequential(
            nn.Conv1d(64, 64, 3, 1),
            nn.Conv1d(64, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, 3, 1),
            nn.Conv1d(32, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            )

        self.gender_classifier = nn.Sequential(
            nn.Linear(1920, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_genders),
        )
        self.age_classifier = nn.Sequential(
            nn.Linear(1920, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_ages),
        )

    def forward(self, x):

        x = self.conv2d(x)
        x = torch.squeeze(x, dim=2)
        x = self.conv1d(x)
        x = torch.flatten(x, 1)
        gender_output = self.gender_classifier(x)
        age_output = self.age_classifier(x)

        # Make sure we don't have nans
        assert not torch.isnan(gender_output).any()
        assert not torch.isnan(age_output).any()

        outputs = {
            'gender': gender_output,
            'age': age_output,
        }

        return outputs
