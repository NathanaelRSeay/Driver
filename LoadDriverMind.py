
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


model = BinaryCNN()
model.load_state_dict(torch.load("DriverMind_weights.pth"))
model.to(device)
model.eval()