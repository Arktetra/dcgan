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
        
    Example usage:
        ```
        model.apply(weights_init)
        ```
    """
    
    classname = module.__Class__.__name__
    
    if classname.find('Conv') != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)
        
class Generator(nn.Module):
    """Creates a Generator for a DCGAN.

    Attributes:
        ngpu (int): the number of GPUs available
        main: the module for the generator
    
    Example usage:
        ```
        netG = Generator(nz = 100, ngf = 64, nc = 3, ngpu = 1)
        ```
    """
    
    def __init__(self, nz: int, ngf: int, nc: int, ngpu: int) -> None:
        """
        Args:
            nz (int): the size of z latent vector
            ngf (int): the number of feature maps
            nc (int): the number of channels in the training images
        """
        
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias = False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace = True),
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace = True),
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace = True),
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace = True),
            
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias = False),
            nn.Tanh()
        )
        
    def forward(self, input: torch.Tensor):
        return self.main(input)
    
class Discriminator(nn.Module):
    """Creates a Disciminator for a DCGAN.

    Args:
        ngpu (int): the number of GPUs available
        main: the module for the discriminator
    
    Example usage:
        ```
        netD = Discriminator(nc = 3, ndf = 64, ngpu = 1)
        ```
    """
    
    def __init__(self, nc: int, ndf: int, ngpu: int) -> None:
        """
        Parameters:
            nc (int): the number of channels in the training images
            ndf (int): the number of feature maps
            ngpu (int): the number of GPUs available
        """
        
        super(Discriminator, self).__init__()
        self.ngpu = ngpu 
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias = False),
            nn.Sigmoid()
        )
        
    def forward(self, input: torch.Tensor):
        return self.main(input)