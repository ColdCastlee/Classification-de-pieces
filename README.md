# 🪙 Coin Classification with AlexNet

This project implements a Convolutional Neural Network (CNN) using **AlexNet** to classify coin images from the DL4CV Kaggle competition.

---

## 📌 Overview

The goal of this project is to classify coin images into **315 different classes**, where each class represents a unique combination of:

- Country
- Currency
- Denomination

This is a challenging computer vision task due to:

- Large number of classes
- Subtle visual differences between coins
- Variations in lighting, angle, and condition

---

## 📂 Project Structure

```
.
├── dataset_coin.py       # Custom PyTorch Dataset
├── train_alexnet.py      # Training script
├── predict_test.py       # Inference script (generate submission)
├── README.md             # Project documentation
└── kaggle/               # Dataset (not included in repo)
```

---

## 🧠 Model

- Architecture: **AlexNet**
- Pretrained: **ImageNet**
- Output classes: **315**
- Framework: **PyTorch**

### Key Modifications

- Replaced final fully connected layer:
  ```python
  model.classifier[6] = nn.Linear(4096, 315)
  ```

- Fine-tuning strategy:
  - Freeze feature extractor initially
  - Then unfreeze for full training

---

## ⚙️ Training

### Run training

```bash
python train_alexnet.py
```

### Training configuration

- Batch size: 32  
- Learning rate: 1e-4  
- Optimizer: Adam  
- Loss: CrossEntropyLoss  
- Epochs: 25  

### Data augmentation

- Random resized crop  
- Rotation  
- Color jitter  
- Horizontal flip (optional)  

---

## 📊 Results

### Validation Performance

| Model                          | Accuracy |
|--------------------------------|----------|
| AlexNet (from scratch)         | 43%      |
| AlexNet (pretrained + augment) | **79.56%** |

### Kaggle Performance

- Public Score: **0.82059**
- Private Score: **0.75819**

---

## 🔮 Inference

Generate predictions for test set:

```bash
python predict_test.py
```

Output:

```
submission_test.csv
```

Format:

```
Id,Class
1234,1 Cent,Australian dollar,australia
...
```

---

## 📁 Dataset

Source:  
👉 Kaggle – DL4CV Coin Classification

- ~10,368 training images  
- ~1,288 test images  
- 315 classes  

⚠️ Dataset is not included in this repository.

---

## 🚀 Improvements

Possible improvements include:

- Using more advanced architectures (ResNet, EfficientNet)
- Handling class imbalance
- Applying Test-Time Augmentation (TTA)
- Increasing input image resolution
- Model ensembling

---

## ⚠️ Limitations

- AlexNet is relatively old architecture
- Difficult to distinguish visually similar coins
- Performance gap between public/private leaderboard

---

## 📌 Notes

- Model weights (.pth) are not included due to size limitations
- Dataset is not included

---

## 👨‍💻 Author

FENG Xiangrui  
Université Paris Cité – M1 Informatique VMI  

---

## 📜 License

This project is licensed under the MIT License.
