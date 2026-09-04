import torch.nn as nn
import torch.optim as optim

from config import (
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    MODEL_SAVE_PATH
)

from data_loader import get_dataloaders
from model import create_model
from train import train_one_epoch, validate
from utils import save_model


def main():

    print(f"Using device: {DEVICE}")

    # Load data
    train_loader, val_loader = get_dataloaders()

    # Create model
    model = create_model()
    model = model.to(DEVICE)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("Ready to train!")

    # Training
    for epoch in range(NUM_EPOCHS):

        print(
            f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]"
        )

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE
        )

        val_accuracy = validate(
            model,
            val_loader,
            DEVICE
        )

        print(
            f"Loss: {train_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

    # Save model
    save_model(
        model,
        MODEL_SAVE_PATH
    )

    print("\nTraining finished!")


if __name__ == "__main__":
    main().p