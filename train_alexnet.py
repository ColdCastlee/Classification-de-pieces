from pathlib import Path
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split
from dataset_coin import CoinDataset


# =========================
# 1. 路径设置
# =========================
DATA_DIR = Path(r"G:\Vmi\S2\Réseaux de neurones\tpKaggle\kaggle")
train_csv = DATA_DIR / "train.csv"
train_dir = DATA_DIR / "train"

# 模型保存路径
save_path = Path(r"G:\Vmi\S2\Réseaux de neurones\tpKaggle\best_alexnet.pth")


# =========================
# 2. 超参数
# =========================
batch_size = 32
num_epochs = 25
learning_rate = 1e-4
val_ratio = 0.2
random_seed = 42


# =========================
# 3. 设备
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前设备:", device)


# =========================
# 4. 图像预处理
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# 5. 数据集
# =========================
full_dataset = CoinDataset(train_csv, train_dir, transform=transform)
num_classes = len(full_dataset.class_to_idx)

print("总样本数:", len(full_dataset))
print("类别数:", num_classes)

train_size = int((1 - val_ratio) * len(full_dataset))
val_size = len(full_dataset) - train_size

generator = torch.Generator().manual_seed(random_seed)
train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=generator
)

print("训练集大小:", len(train_dataset))
print("验证集大小:", len(val_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)


# =========================
# 6. 模型
# =========================
model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)

in_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(in_features, num_classes)

model = model.to(device)

print("\n模型最后一层:")
print(model.classifier[6])


# =========================
# 7. 损失函数和优化器
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# =========================
# 8. 训练与验证函数
# =========================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, dim=1)

        batch_size_now = labels.size(0)
        running_loss += loss.item() * batch_size_now
        running_correct += (preds == labels).sum().item()
        total_samples += batch_size_now

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    return epoch_loss, epoch_acc


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, dim=1)

            batch_size_now = labels.size(0)
            running_loss += loss.item() * batch_size_now
            running_correct += (preds == labels).sum().item()
            total_samples += batch_size_now

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    return epoch_loss, epoch_acc


# =========================
# 9. 正式训练
# =========================
best_val_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())

history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

print("\n开始训练...\n")

for epoch in range(num_epochs):
    print(f"Epoch [{epoch + 1}/{num_epochs}]")

    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )

    val_loss, val_acc = validate_one_epoch(
        model, val_loader, criterion, device
    )

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save({
            "model_state_dict": best_model_wts,
            "class_to_idx": full_dataset.class_to_idx,
            "num_classes": num_classes
        }, save_path)
        print(f"已保存更好的模型到: {save_path}")

    print("-" * 50)

print("\n训练完成。")
print(f"最佳验证集准确率: {best_val_acc:.4f}")

# 把最佳权重重新载入
model.load_state_dict(best_model_wts)