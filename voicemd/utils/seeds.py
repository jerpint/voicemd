import random
import numpy as np
import torch

def set_seeds(seed):
    '''Set seeds for reproducibility.'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
