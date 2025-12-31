# ✈️ Airplane Hull Defect Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A Computer Vision pipeline designed to detect and classify structural defects on airplane hulls. This system utilizes deep learning to distinguish between critical **Cracks** and surface **Dents**, aiding in automated predictive maintenance.

## 🧠 Project Architecture

The project implements a competitive model selection pipeline:
1.  **Custom CNN:** A lightweight, highly efficient Convolutional Neural Network designed for texture analysis.
2.  **Transfer Learning (VGG16):** A heavy, pre-trained feature extractor adapted for defect classification.

The system automatically trains both, compares their validation accuracy, and deploys the superior model for inference.

## 📂 Dataset Structure
The model is trained on a categorized dataset of hull defects:
* **Crack:** Structural fractures requiring immediate attention.
* **Dent:** Surface depressions requiring monitoring.

## 🚀 How to Run

### 1. Prerequisites
Ensure you have Python installed. Install dependencies:
```bash
pip install tensorflow numpy matplotlib
```

### 2. Full Pipeline Execution 
To run the entire lifecycle (Train -> Compare -> Test -> Demo):
```bash
python main.py
```

### 3. Interactive Demo
To strictly test the trained model on random test images without retraining: 
```bash
python predict.py
```

## 📊 Pipeline Modules


## 📈 Results
    - Visualizations: Training accuracy and loss graphs are automatically saved to the visualizations/ folder.

    - Performance: In testing, the Custom CNN demonstrated superior performance on texture-based defect recognition compared to the frozen VGG16 baseline.

## 👤 Author
(Rishikesh Naware)[https://www.linkedin.com/in/rishikesh-naware-9a18ab1a8/]
