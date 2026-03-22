from pathlib import Path
import random
import pandas as pd
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset


class CoinDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, class_to_idx=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform

        if class_to_idx is None:
            classes = sorted(self.df["Class"].unique())
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.extensions = [".jpg", ".jpeg", ".png", ".webp"]

    def __len__(self):
        return len(self.df)

    def _find_image_path(self, img_id):
        img_id = str(img_id)
        for ext in self.extensions:
            img_path = self.img_dir / f"{img_id}{ext}"
            if img_path.exists():
                return img_path
        raise FileNotFoundError(f"找不到图片: {img_id}")

    def __getitem__(self, idx):
        while True:
            row = self.df.iloc[idx]

            img_id = row["Id"]
            class_name = row["Class"]

            try:
                img_path = self._find_image_path(img_id)
                image = Image.open(img_path).convert("RGB")
                label = self.class_to_idx[class_name]

                if self.transform is not None:
                    image = self.transform(image)

                return image, label

            except (UnidentifiedImageError, OSError, FileNotFoundError):
                idx = random.randint(0, len(self.df) - 1)