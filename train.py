import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import os
from tqdm import tqdm
# --- تنظیمات ---
data_dir = "EuroSAT_RGB"  # فولدر اصلی دیتا
batch_size = 32
num_epochs = 10
learning_rate = 0.001
train_val_split = 0.8  # 80% آموزش و 20% اعتبارسنجی
num_classes = 10  # 10 کلاس

# --- تبدیل‌ها ---
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # تغییر سایز تصاویر به 64x64
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# --- بارگذاری داده‌ها ---
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
print("dataset added")
# تقسیم به train و validation
train_size = int(train_val_split * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print("data loaded")
# --- مدل ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=True)  # استفاده از ResNet18 پیش‌آموزش‌دیده
model.fc = nn.Linear(model.fc.in_features, num_classes)  # تغییر لایه آخر برای 10 کلاس
model = model.to(device)

# --- Loss و Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print("ready to train")
# --- حلقه آموزش ---
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        loop.set_postfix(loss=loss.item())
    
    epoch_loss = running_loss / len(train_loader.dataset)
    
    # اعتبارسنجی
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_acc = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}")

# --- ذخیره مدل ---
torch.save(model.state_dict(), "resnet18_eurosat.pth")
print("Training finished and model saved!")
