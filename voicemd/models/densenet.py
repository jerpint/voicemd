import torch
import torchvision.models as models

def densenet121(hyper_params=None):

    densenet = models.densenet121(pretrained=True)
    densenet.classifier = torch.nn.Linear(in_features=1024, out_features=512)
    densenet = torch.nn.Sequential(densenet, torch.nn.Linear(in_features=512, out_features=1))
    return densenet
