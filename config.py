import torch

DATA_DIR = "EuroSAT_RGB"

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

TRAIN_VAL_SPLIT = 0.8
NUM_CLASSES = 10

IMAGE_SIZE = 64

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_SAVE_PATH = "resnet18_eurosat.pth"