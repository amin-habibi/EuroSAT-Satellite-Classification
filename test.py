import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os


path = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
print(path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 10

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# model.load_state_dict(torch.load("d:/1-projects/EuroSAT_RGB/resnet18_eurosat.pth", map_location=device))
model.load_state_dict(torch.load(path + "/resnet18_eurosat.pth", map_location=device))

model = model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])


from torchvision.datasets import ImageFolder

# dataset = ImageFolder(path + "/EuroSAT_RGB")
# class_names = dataset.classes

class_names = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # [1, 3, 64, 64]
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return class_names[predicted.item()]


img_path = path + "/test.jpg"
pred = predict_image(img_path)
print("Predicted class:", pred)

if pred in ["Herbaceous Vegetation", "Highway", "Pasture"]:
    print("احتمال تغییر کاربری وجود دارد")
else:
    print(" احتمال تغییر کاربری وجود ندارد ")
