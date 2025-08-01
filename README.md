# image-_classificaton_for_disease_detection
# 🩺 Medical Image Classification for Disease Detection

## 🧠 Project Overview

This project focuses on classifying **chest X-ray images** into two categories: **NORMAL** and **PNEUMONIA** using **Convolutional Neural Networks (CNN)** in PyTorch. This is a deep learning application in the healthcare domain, aimed at assisting in early and efficient disease detection using medical images.

---

## 🎯 Objective

To build a deep learning model that can:
- Accurately detect **pneumonia** from chest X-ray images.
- Generalize well on unseen data using data augmentation and regularization.
- Evaluate the model with robust metrics: **accuracy**, **precision**, and **recall**.

---

## 📁 Dataset

**Dataset Used**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

**Structure:**
chest_xray/
├── train/
│ ├── NORMAL/
│ └── PNEUMONIA/
├── val/
│ ├── NORMAL/
│ └── PNEUMONIA/
├── test/
├── NORMAL/
└── PNEUMONIA/

---

## ⚙️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Programming language |
| PyTorch | Deep learning framework |
| torchvision | Image loading and transformations |
| matplotlib | Plotting & visualization |
| scikit-learn | Performance metrics |
| OpenCV (optional) | Image processing (if needed) |

---

## 🧩 Project Workflow

### 1. **Data Preprocessing**
- Resize all images to `224x224`.
- Normalize pixel values.
- Augment training data using horizontal flips and random rotation.

### 2. **Data Splitting**
- Train, Validation, and Test datasets are already structured in folders.
- Loaded using `torchvision.datasets.ImageFolder`.

### 3. **Model Building**
- Custom CNN with two convolutional layers and two fully connected layers.
- Trained using **CrossEntropyLoss** and **Adam optimizer**.
- Uses **ReLU activation** and **MaxPooling**.

### 4. **Training**
- Trained for 10 epochs with data augmentation.
- Loss and accuracy monitored during training.

### 5. **Evaluation**
- Evaluated on test data using:
  - **Accuracy**
  - **Precision**
  - **Recall**
- Classification Report and Confusion Matrix plotted.

### 6. **Visualization**
- Misclassified images shown with predicted vs actual labels.

---

## 📊 Sample Results

| Metric | Value |
|--------|-------|
| Accuracy | ~92% |
| Precision | ~91% |
| Recall | ~93% |

> *(Note: Results may vary based on data split, epochs, model structure, etc.)*

---

## 🖼️ Sample Output

![Prediction Sample](sample_output.png)

---

## 🚀 Future Enhancements

- Use **Transfer Learning** (e.g., ResNet18, EfficientNet).
- Add **GUI or Web App** using Streamlit or Flask.
- Use **Grad-CAM** to visualize model attention.
- Expand dataset with multi-class diseases.

---

## 📌 How to Run

1. **Clone this repo**
```bash
git clone https://github.com/your-username/medical-image-classification.git
cd medical-image-classification


output:
| Class                | Precision | Recall | F1-Score | Support |
| -------------------- | --------- | ------ | -------- | ------- |
| **NORMAL**           | 0.89      | 0.92   | 0.91     | 234     |
| **PNEUMONIA**        | 0.95      | 0.88   | 0.91     | 390     |
| **Overall Accuracy** | —         | —      | **91%**  | 624     |

