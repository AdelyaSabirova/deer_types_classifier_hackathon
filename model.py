import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

def get_model(device):
    model = efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 3)  # 3 класса
    model = model.to(device)
    return model
