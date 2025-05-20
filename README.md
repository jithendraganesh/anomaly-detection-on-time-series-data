
# 🌡️ Anomaly Detection Using LSTM and STOC

This project explores and compares two deep learning approaches for anomaly detection in time-series data:
- A traditional **LSTM** model.
- A modern **Stacked Transformer with 1D CNN (STOC)** based on recent research.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

---

## 📖 Overview

This notebook performs a **comparative study of two architectures**:
- **LSTM**: A classical RNN-based model well-suited for sequence modeling and anomaly detection.
- **STOC**: A hybrid model combining transformer blocks for representation learning and 1D CNNs for local temporal pattern extraction.

The goal is to evaluate how modern attention-based architectures compare with traditional sequential models in the context of time-series anomaly detection.

---

## 📊 Dataset

The dataset used in this study is from Kaggle:
- **Name**: Weather Analysis Dataset
- **File**: `climate_data.csv`
- **Features**: Includes temperature, humidity, pressure, and other climate indicators.
- **Target**: Anomaly score based on reconstructed prediction vs actual values.

---

## 🧠 Models

### 1. LSTM Autoencoder
- Used as a benchmark baseline.
- Encodes the time series into a latent space and reconstructs it.
- High reconstruction error signals potential anomalies.

### 2. STOC (Stacked Transformer + 1D CNN)
- Implements multi-head self-attention to capture global temporal dependencies.
- Combines with 1D CNN layers for localized feature extraction.
- More complex, inspired by recent research on hybrid architectures.

---

## ⚙️ Installation

Make sure you are running in a Kaggle environment or install the following locally:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

---

## 🚀 Usage

1. Clone this repository or run the notebook on [Kaggle](https://www.kaggle.com/).
2. Load the dataset:
   - Ensure the `climate_data.csv` is available in the input folder.
3. Run all cells in the notebook.
4. View loss plots, reconstruction errors, and anomaly flags.

---

## 📈 Results

The notebook provides:
- **Visualization of time-series anomalies**
- **Model loss curves**
- **Precision, Recall, F1-score** metrics
- **Comparison of anomaly detection capability** between LSTM and STOC

Key findings:
- LSTM performs well on simple temporal patterns.
- STOC offers improved anomaly localization and robustness to complex data.

---

## 📚 References

- Research Paper: *Time-Series Anomaly Detection with Stacked Transformer Representations and 1D Convolutional Network*
- Kaggle Dataset: [Weather Analysis](https://www.kaggle.com/datasets)
- TensorFlow Documentation
- PyTorch Transformers

---

## 🙌 Acknowledgements

Developed as part of a deep learning project focused on time-series anomaly detection using both classical and state-of-the-art methods.
