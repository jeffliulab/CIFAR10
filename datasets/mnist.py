import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

DATASET_DIR = "datasets/downloads"

def get_loaders(batch_size: int = 128,
                val_fraction: float = 0.2):
    transform = transforms.ToTensor()
    
    full_train = datasets.MNIST(DATASET_DIR, train=True, download=True, transform=transform)
    test_ds    = datasets.MNIST(DATASET_DIR, train=False, download=False, transform=transform)
    
    # Use the given val_fraction to define train_size and val_size
    train_size = int((1 - val_fraction) * len(full_train))
    val_size   = len(full_train) - train_size
    generator  = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=generator)

    # Wrap the training, validation and test datasets with DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
    