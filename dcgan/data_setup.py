"""
Contains functionality for creating PyTorch DataLoaders for
generative adversarial networks
"""

import os

from torch.utils.data import DataLoader

from torchvision.transforms import v2
from torchvision import datasets

NUM_WORKERS = os.cpu_count

def create_dataloaders(
    root: str,
    transform: v2.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS
) -> DataLoader:
    """Creates Dataloader for Generative Adversarial Networks.
    
    Takes in a root directory path and turns them into PyTorch
    Datasets and then into PyTorch DataLoaders.

    Args:
        root (str): Path to training data.
        transform (v2.Compose): torchvision transforms to perform on training data.
        batch_size (int): Number of samples per batch in DataLoader
        num_workers (int, optional): An integer for number of workers in DataLoader. Defaults to NUM_WORKERS.
        
    Returns:
        A DataLoader.
        
    Example usage:
        dataloader = create_dataloader(
            root = path/to/data,
            transform = some_transform,
            batch_size = 32,
            num_workers = 2
        )
    """
    
    # Use ImageFolder to create dataset
    data = datasets.ImageFolder(
        root = root,
        transform = transform,
    )
    
    # Turn images into data loaders
    dataloader = DataLoader(
        data,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers
    )
    
    return dataloader
    