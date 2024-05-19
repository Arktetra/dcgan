"""
Contains PyTorch model code to instantiate a Generative Adversarial Network
"""

import torch 
from torch import nn

def weights_init(
    module: nn.Module
):
    """A function to be applied on modules of PyTorch.
    
    Takes in a module and intializes its Conv and BatchNorm
    layer weights except at last layer.

    Args:
        module (nn.Module): a module of PyTorch
    """
    classname = module.__Class__.__name__
    
    if classname.find('Conv') != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)
        
