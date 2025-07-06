
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def load_model(model, optimizer, path="checkpoint.pth", device='cpu'):
    """
    Load model and optimizer state from a file.
    Returns: model, optimizer, start_epoch, loss
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"✅ Model loaded from: {path} (Epoch {start_epoch})")
    return model.to(device), optimizer, start_epoch, loss