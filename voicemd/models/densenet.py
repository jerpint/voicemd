import torch
import torchvision.models as models
from torchvision.models import densenet


def densenet121(hyper_params=None):

    densenet = models.densenet121(pretrained=True)
    densenet.classifier = torch.nn.Linear(in_features=1024, out_features=1000)
    densenet = torch.nn.Sequential(densenet, torch.nn.Linear(in_features=1000, out_features=1))
    return densenet


def densenet_small(hyper_params=None):

    model = densenet._densenet('densenet_small', 24, (6, 12, 24), 32, pretrained=False, progress=True, num_classes=1)

    return model
