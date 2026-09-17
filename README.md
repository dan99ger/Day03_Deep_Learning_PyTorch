# 🧠 Day 03: Deep Learning & Custom PyTorch Training Loop

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

A hands-on implementation of an Artificial Neural Network (ANN) using PyTorch. This project focuses on building low-level Dataset pipelines, DataLoader batching, and a custom training loop with explicit backpropagation steps.

---

## 🛠️ Key Features

* **Custom Pipeline:** Built-in PyTorch `Dataset` and `DataLoader` classes for batch processing.
* **Network Architecture:** Multi-layer Perceptron (MLP) incorporating `BatchNorm1d`, `ReLU`, and `Dropout` layers.
* **Manual Training Loop:** Explicit gradient zeroing, forward pass, loss computation, backpropagation, and optimization step execution.
* **Loss Tracking:** Generates and exports training vs. validation loss curve graphs.

---

## 📂 Repository Structure

```text
Day03_Deep_Learning_PyTorch/
├── artifacts/
│   ├── pytorch_ann_model.pth    # Model weights state dict
│   └── loss_curve.png           # Loss evaluation plot
├── nn_from_scratch.py           # PyTorch architecture & training loop script
├── .gitignore
└── README.md


🚀 How to Run
1. Setup Environment
Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install torch pandas numpy scikit-learn matplotlib
2. Execute PyTorch Training Loop
Bash
python nn_from_scratch.py
Part of the 7-Day Machine Learning Engineering Challenge.
