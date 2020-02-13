import torch
import torchvision.models as models

def densenet121(hyper_params=None):

    ### TODO, figure out if we want to pop the last layer
    densenet = models.densenet121(pretrained=True)
    #  densenet._modules.pop('classifier')
    densenet = torch.nn.Sequential(densenet, torch.nn.Linear(in_features=1000, out_features=1))

    return densenet
