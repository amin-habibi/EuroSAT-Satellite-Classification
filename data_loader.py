from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from config import (
    DATA_DIR,
    BATCH_SIZE,
    TRAIN_VAL_SPLIT,
    IMAGE_SIZE,
    MEAN,
    STD
)


def get_transforms():

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])


def get_dataloaders():

    transform = get_transforms()

    dataset = datasets.ImageFolder(
        root=DATA_DIR,
        transform=transform
    )

    print(f"Dataset loaded: {len(dataset)} images")

    train_size = int(
        TRAIN_VAL_SPLIT * len(dataset)
    )

    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    return train_loader, val_loader