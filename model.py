import torch.nn as nn
from torchvision import models

from config import NUM_CLASSES


def create_model():

    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    return model