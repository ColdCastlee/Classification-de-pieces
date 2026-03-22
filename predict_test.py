from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


# =========================
# 1. 路径设置
# =========================
DATA_DIR = Path(r"G:\Vmi\S2\Réseaux de neurones\tpKaggle\kaggle")
TEST_CSV = DATA_DIR / "test.csv"
TEST_DIR = DATA_DIR / "test"

MODEL_PATH = Path(r"G:\Vmi\S2\Réseaux de neurones\tpKaggle\best_alexnet.pth")
SUBMISSION_PATH = Path(r"G:\Vmi\S2\Réseaux de neurones\tpKaggle\submission_test.csv")


# =========================
# 2. 设备
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前设备:", device)


# =========================
# 3. 测试集预处理
# =========================
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# 4. 测试集 Dataset
# =========================
class CoinTestDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.extensions = [".jpg", ".jpeg", ".png", ".webp"]

    def __len__(self):
        return len(self.df)

    def _find_image_path(self, img_id):
        img_id = str(img_id)
        for ext in self.extensions:
            img_path = self.img_dir / f"{img_id}{ext}"
            if img_path.exists():
                return img_path
        raise FileNotFoundError(f"找不到测试图片: {img_id}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["Id"]

        try:
            img_path = self._find_image_path(img_id)
            image = Image.open(img_path).convert("RGB")

            if self.transform is not None:
                image = self.transform(image)

            return image, str(img_id)

        except (UnidentifiedImageError, OSError, FileNotFoundError):
            print(f"警告：测试图片读取失败: {img_id}")
            dummy = torch.zeros(3, 224, 224)
            return dummy, str(img_id)


# =========================
# 5. 载入模型 checkpoint
# =========================
checkpoint = torch.load(MODEL_PATH, map_location=device)

class_to_idx = checkpoint["class_to_idx"]
num_classes = checkpoint["num_classes"]
idx_to_class = {v: k for k, v in class_to_idx.items()}

print("类别数:", num_classes)


# =========================
# 6. 重建 AlexNet 模型
# =========================
model = models.alexnet(weights=None)
in_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(in_features, num_classes)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

print("模型加载完成")


# =========================
# 7. DataLoader
# =========================
test_dataset = CoinTestDataset(TEST_CSV, TEST_DIR, transform=test_transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

print("测试样本数:", len(test_dataset))


# =========================
# 8. 预测
# =========================
all_ids = []
all_preds = []

with torch.no_grad():
    for images, img_ids in test_loader:
        images = images.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().tolist()

        pred_classes = [idx_to_class[p] for p in preds]

        all_ids.extend(img_ids)
        all_preds.extend(pred_classes)


# =========================
# 9. 保存 submission csv
# =========================
submission_df = pd.DataFrame({
    "Id": all_ids,
    "Class": all_preds
})

submission_df.to_csv(SUBMISSION_PATH, index=False, encoding="utf-8")

print(f"\n预测完成，文件已保存到: {SUBMISSION_PATH}")
print("\n前5行预览:")
print(submission_df.head())