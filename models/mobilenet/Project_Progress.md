# 🩺 Explainable AI for Pneumonia Detection using MobileNetV2

## 📌 Project Overview

This project focuses on building a deep learning-based medical image classification system to detect **Pneumonia** from Chest X-ray images.

The primary objective of this phase is:

* Train a pretrained **MobileNetV2** model
* Fine-tune it on a Chest X-ray dataset
* Evaluate its performance using standard classification metrics
* Prepare the model for integration with Explainable AI (XAI) techniques

---

## 📂 Dataset Description

**Dataset Name:** Chest X-Ray Images (Pneumonia)
**Total Images:** 5,863
**Classes:**

* NORMAL
* PNEUMONIA

### Folder Structure

```
chest_xray/
│
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

The dataset contains pediatric chest X-rays categorized into Normal and Pneumonia classes.

---

## 🧠 Model Used

### 🔹 MobileNetV2

* Pretrained on ImageNet
* Lightweight CNN architecture
* Uses inverted residual blocks
* Efficient and suitable for deployment
* Modified final classifier layer for binary classification

```python
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, 2)
```

---

## ⚙️ Data Preprocessing

All images were resized and normalized before training.

### Transformations Applied:

* Resize to 224 × 224
* Convert grayscale to 3-channel RGB
* Convert to tensor
* Normalize with mean = 0.5 and std = 0.5

```python
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
```

---

## 🏋️ Training Configuration

* Loss Function: CrossEntropyLoss
* Optimizer: Adam
* Learning Rate: 0.0001
* Batch Size: 32
* Epochs: 10
* Device: GPU (Google Colab)

---

## 📊 Model Evaluation

The trained model was evaluated on the test dataset.

### Metrics Used:

* Accuracy
* Precision
* Recall
* F1-Score

### 📈 Test Accuracy

> **Test Accuracy: 85%**

The model achieved high accuracy in distinguishing between Normal and Pneumonia cases.

---

## 💾 Model Saving

The trained model was saved in `.pth` format for future deployment and integration with XAI methods.

```python
torch.save(model.state_dict(), "mobilenet_pneumonia.pth")
```

This allows:

* Reloading model for inference
* Integration with backend API
* Application of XAI techniques

---

## ✅ Work Completed So Far

✔ Dataset preprocessing
✔ Model selection (MobileNetV2)
✔ Transfer learning implementation
✔ Training on Chest X-ray dataset
✔ Evaluation and accuracy calculation
✔ Model saved for further explainability analysis

---

## 🔜 Next Phase

In the next phase, the trained model will be integrated with Explainable AI techniques:

* Grad-CAM++
* LIME
* SHAP

These methods will help interpret model predictions and highlight regions responsible for classification decisions.

---

## 👨‍💻 Author

Savara Pradeep
Major Project – Deep Learning & Explainable AI
2026

---
