# 🔬 Dermo-Scope: Real-Time Skin Disease Detection System

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange?logo=tensorflow)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Dermo-Scope** is an end-to-end deep learning system that classifies skin lesions into 7 categories using the HAM10000 dataset and provides **real-time webcam analysis** with **Grad-CAM explainability**.

---

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset Setup](#dataset-setup)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Running the Web App](#running-the-web-app)
- [Model Architecture](#model-architecture)
- [Grad-CAM Explainability](#grad-cam-explainability)
- [Risk Classification](#risk-classification)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎥 **Live Webcam** | Real-time skin lesion detection via WebRTC |
| 🖼️ **Image Upload** | Drag-and-drop image inference |
| 🧠 **Grad-CAM** | Heatmap highlights regions influencing prediction |
| 🚦 **Risk Indicators** | Red (high risk) / Green (low risk) color coding |
| 📊 **Top-3 Predictions** | Confidence bars for top 3 classes |
| 📈 **Training Plots** | Accuracy/loss curves and confusion matrix |
| 🔬 **7-Class Detection** | Full HAM10000 classification |

---

## 🗂️ Project Structure

```
major_project/
├── data_tools/
│   └── 01_organize_data.py     # Organizes HAM10000 into class folders
├── model_training/
│   ├── 02_train_model.py       # MobileNetV2 training script
│   ├── saved_model.h5          # Trained model (generated)
│   ├── training_history.png    # Accuracy/Loss curves (generated)
│   ├── confusion_matrix.png    # Confusion matrix (generated)
│   └── classification_report.txt
├── app/
│   ├── main.py                 # Streamlit web application
│   └── utils.py                # Grad-CAM, preprocessing, inference
├── raw_data/                   # Place HAM10000 images + CSV here
├── organized_data/             # Auto-generated class folders
│   ├── train/                  # 80% split
│   │   ├── mel/
│   │   ├── nv/
│   │   └── ... (7 classes)
│   └── val/                    # 20% split
│       └── ...
├── requirements.txt
└── README.md
```

---

## 📦 Dataset Setup

### Step 1 – Download HAM10000

Download the HAM10000 dataset from Kaggle:  
👉 https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection

You need:
- `HAM10000_metadata.csv`
- All `.jpg` images (from both image parts)

### Step 2 – Place files in `raw_data/`

```
major_project/
└── raw_data/
    ├── HAM10000_metadata.csv
    ├── ISIC_0024306.jpg
    ├── ISIC_0024307.jpg
    └── ... (all ~10,015 images)
```

### Step 3 – Run the organization script

```bash
cd major_project
python data_tools/01_organize_data.py
```

This will:
- Read the metadata CSV
- Split images 80% train / 20% validation (stratified)
- Organize into `organized_data/train/<class>/` and `organized_data/val/<class>/`
- Verify compatibility with Keras `flow_from_directory`

---

## ⚙️ Installation

### Prerequisites
- Python 3.9 or higher
- pip

### Install dependencies

```bash
cd major_project
pip install -r requirements.txt
```

> **GPU Support**: Install `tensorflow-gpu` instead of `tensorflow` for significantly faster training.

---

## 🏋️ Training the Model

After organizing the dataset:

```bash
cd major_project
python model_training/02_train_model.py
```

### Training Configuration

| Parameter | Value |
|---|---|
| Base Model | MobileNetV2 (ImageNet) |
| Input Size | 224 × 224 × 3 |
| Optimizer | Adam (lr=0.0001) |
| Loss | Categorical Crossentropy |
| Batch Size | 32 |
| Max Epochs | 20 |
| Early Stopping | patience=5 |
| LR Reduction | factor=0.5, patience=3 |

### Expected Output

```
organized_data/train  →  ~8,012 images
organized_data/val    →  ~2,003 images

Epoch 1/20 — val_accuracy: 0.62
Epoch 5/20 — val_accuracy: 0.72
...
Final Val Accuracy: ~70–85%
```

Generated files:
- `model_training/saved_model.h5`
- `model_training/training_history.png`
- `model_training/confusion_matrix.png`
- `model_training/classification_report.txt`

---

## 🚀 Running the Web App

```bash
cd major_project
streamlit run app/main.py
```

Open your browser at: **http://localhost:8501**

### Demo Mode

If `saved_model.h5` is not found, the app runs in **demo mode** with random predictions so you can explore the UI.

---

## 🧬 Model Architecture

```
Input (224×224×3)
    │
    ▼
MobileNetV2 (imagenet, frozen)
    │
    ▼
GlobalAveragePooling2D
    │
    ▼
Dropout(0.3)
    │
    ▼
Dense(128, relu)
    │
    ▼
Dropout(0.3)
    │
    ▼
Dense(7, softmax)  →  [akiec, bcc, bkl, df, mel, nv, vasc]
```

---

## 🧠 Grad-CAM Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights which pixels in the image most influenced the model's prediction.

**How it works:**

1. Forward pass through the model
2. Compute gradients of the target class score with respect to `Conv_1` (last conv layer of MobileNetV2)
3. Pool gradients spatially → importance weights
4. Weight feature maps → class activation map
5. Upsample to 224×224
6. Overlay red heatmap on the original image

**Triggered automatically** for High Risk predictions (mel, bcc, akiec).

---

## 🚦 Risk Classification

| Class | Full Name | Risk |
|---|---|---|
| `mel` | Melanoma | 🔴 High |
| `bcc` | Basal Cell Carcinoma | 🔴 High |
| `akiec` | Actinic Keratoses / Intraepithelial Carcinoma | 🔴 High |
| `nv` | Melanocytic Nevi | 🟢 Low |
| `bkl` | Benign Keratosis | 🟢 Low |
| `df` | Dermatofibroma | 🟢 Low |
| `vasc` | Vascular Lesions | 🟢 Low |

---

## ⚠️ Disclaimer

> **Dermo-Scope is an educational project and is NOT a medical diagnostic tool.**  
> Never use this system to self-diagnose or replace professional medical advice.  
> Always consult a qualified dermatologist for any skin concerns.

---

## 📄 License

MIT License – see [LICENSE](LICENSE) for details.
