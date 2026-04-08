# Human Pose Estimation with Deep Learning

Comparative study of deep learning architectures (CNN vs Transformer) for human pose estimation using the MPII dataset.

---

## 🧠 Overview

This project addresses the problem of **human pose estimation**, which consists of predicting keypoints of the human body from images.

We focus on detecting three keypoints:
- Head
- Right wrist
- Left wrist

Two fundamentally different approaches are implemented and compared:

- **CNN (ResNet50)** → Direct regression of coordinates  
- **Transformer (TokenPose)** → Heatmap-based prediction  

---

## ⚙️ Methodology

The project follows a complete machine learning pipeline:

### 1. Data preprocessing
- Filtering images with a single person
- Selecting relevant keypoints
- Normalizing coordinates
- Train / Validation / Test split

### 2. Model design

#### CNN (ResNet50)
- Transfer learning from ImageNet
- Regression head for coordinate prediction
- Loss: Mean Squared Error (MSE)

#### Transformer (TokenPose)
- Visual tokenization
- Transformer encoder (multi-head attention)
- Heatmap generation (112x112)
- Keypoint extraction via argmax

---

### 3. Training
- Optimizer: Adam
- Data augmentation:
  - Brightness / contrast
  - Gaussian blur
- Learning rate scheduling

---

### 4. Evaluation

Metric used:
- **PCKh@0.3 (Percentage of Correct Keypoints)**

---

## 📊 Results

| Model          | Approach    | Accuracy (PCKh@0.3) |
|----------------|------------|---------------------|
| CNN (ResNet50) | Regression | **82%**             |
| Transformer    | Heatmaps   | **48%**             |

### 🔍 Key Insights

- CNN achieves **higher spatial precision** due to continuous coordinate prediction  
- Transformer captures **semantic regions** but suffers from resolution limitations  
- Heatmap discretization introduces quantization error  

---

## 📁 Project Structure
src/
├── data/ # preprocessing scripts
├── models/ # CNN and Transformer implementations
└── utils/ # visualization and helper functions

notebooks/
└── pose_estimation_experiments.ipynb

data/
└── sample/
├── images/
│ ├── train/
│ ├── val/
│ └── test/
└── labels/

configs/
└── train_config.yaml

results/
└── cnn_vs_transformer/

report/
└── pose_estimation_report.pdf


---

## 📦 Dataset

This project is based on the **MPII Human Pose Dataset**.

Due to size limitations, only a small sample is included in:
data/sample/


To use the full dataset, download it externally and adapt the config file.

---

## ▶️ How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt

jupyter notebook notebooks/pose_estimation_experiments.ipynb


📈 Visualization
Keypoint predictions over images
Transformer heatmaps

Results available in:
results/


⚠️ Model Weights
Model weights are not included due to size limitations.


📄 Report
Detailed report available in:
report/pose_estimation_report.pdf


🧠 Key Takeaways
Regression-based CNNs outperform heatmap-based methods in low-resolution setups
Transformers capture global context but require higher resolution for precision
Model selection depends on accuracy vs computational cost trade-off

