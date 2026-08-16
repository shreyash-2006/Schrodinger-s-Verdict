# Schrödinger's Verdict ⚛️

**Schrödinger's Verdict** is a project to benchmark classical ML algorithms with QML and QKAN. Classial computing has only bits which limits it in data processing after a certian point, however Quantum Computing offers a fundamentally different approach to use **Qubits** for computing. Principles like **Superposition** and **Entanglement** can be exploited for better results. This project aims to use these and study the Quantum advantage over classical ML.

## 📁 Repository Structure

```text
Schrodinger-s-Verdict/
│
├── KAN/
│   ├── Logistic KAN
│   └── PyKAN
│
├── ML/
│   ├── XGBoost
│   ├── SVM
│   └── Logistic Regression
│
├── QML/
│   ├── Quantum Autoencoder
│   ├── VQC
│   └── QSVC
│
└── Mini Project/
    └── Fraud Detection
```

---

## 🧠 KAN

The **KAN** section contains implementations based on **Kolmogorov-Arnold Networks**, exploring KAN-based approaches to machine learning and classification.

### Logistic KAN

A KAN-based implementation for binary classification, combining the KAN approach with a logistic-style classification workflow.

### PyKAN

An implementation using **PyKAN**, including experiments on the Wisconsin Breast Cancer Dataset.

A key feature of KANs is that learnable functions can be associated with the connections between nodes, providing a different formulation from conventional neural networks.

---

## 🤖 Classical Machine Learning

The **ML** section contains classical machine learning implementations that can serve as baselines and comparison models.

### XGBoost

Implementation of **XGBoost** for classification using a gradient-boosted decision-tree ensemble.

### SVM

Implementation of a **Support Vector Machine (SVM)** classifier.

### Logistic Regression

Implementation of **Logistic Regression** for binary classification.

---

## ⚛️ Quantum Machine Learning

The **QML** section contains implementations that explore machine learning with quantum circuits and quantum kernels.

### Quantum Autoencoder

A **Quantum Autoencoder** implementation exploring quantum representation learning and compression.

### VQC — Variational Quantum Classifier

The **VQC** implementation uses a trainable parameterized quantum circuit for classification.

### QSVC — Quantum Support Vector Classifier

The **QSVC** implementation explores quantum-kernel-based support-vector classification.

---

## 💳 Mini Project — Fraud Detection

The **Mini Project** section contains work related to a **fraud detection mini project**.

The project applies machine-learning concepts to the practical problem of identifying potentially fraudulent transactions.

General workflow:

```text
Transaction Data
      ↓
Data Preprocessing
      ↓
Feature Engineering
      ↓
Fraud Detection Model
      ↓
Prediction
      ↓
Fraud / Legitimate
```

---

## 🔬 Project Overview

| Section | Focus | Implementations |
|---|---|---|
| **KAN** | Kolmogorov-Arnold Networks | Logistic KAN, PyKAN |
| **ML** | Classical Machine Learning | XGBoost, SVM, Logistic Regression |
| **QML** | Quantum Machine Learning | Quantum Autoencoder, VQC, QSVC |
| **Mini Project** | Applied Machine Learning | Fraud Detection |

The repository brings these areas together to explore different approaches to machine learning:

```text
Classical Machine Learning
           │
           ▼
    KAN-based Learning
           │
           ▼
 Quantum Machine Learning
           │
           ▼
         QKAN
           │
           ▼
 Real-world Applications
```

---

## 🛠️ Technologies

The implementations use technologies from the Python ML and quantum-computing ecosystem, including:

- Python
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- PyTorch
- PyKAN / KAN
- Qiskit
- Qiskit Machine Learning
- PennyLane
- Matplotlib
- Jupyter Notebook

> Dependency requirements may differ between individual implementations. Check the relevant notebook or source file before running an experiment.

---

## 🚀 Getting Started

Clone the repository:

```bash
git clone https://github.com/shreyash-2006/Schrodinger-s-Verdict.git
cd Schrodinger-s-Verdict
```

Create a virtual environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the packages required by the implementation you want to run.

## 📊 Experiments

The repository provides implementations across three major machine-learning paradigms.

### Classical ML

- Logistic Regression
- SVM
- XGBoost

### KAN

- Logistic KAN
- PyKAN

### QML

- Quantum Autoencoder
- VQC
- QSVC

These implementations can be used to explore differences in:

- Model architecture
- Feature representation
- Training methodology
- Classification workflows
- Classical vs. quantum approaches
- KAN vs. conventional ML approaches

---

## 🎯 Motivation

Machine learning continues to evolve beyond traditional algorithms.

This repository explores three complementary directions:

**Classical ML** — established algorithms that provide strong and interpretable baselines.

**KAN** — an alternative neural-network formulation based on learnable functions associated with network connections.

**QML** — an emerging paradigm combining quantum computing with machine learning.

By placing these implementations in one repository, **Schrödinger's Verdict** provides a practical space for experimenting with different computational approaches to machine learning.

---

## 👨‍💻 Repository

**Schrödinger's Verdict**

[GitHub Repository](https://github.com/shreyash-2006/Schrodinger-s-Verdict)

If you find the project useful, consider ⭐ starring the repository and exploring the different implementations across **ML, KAN, and QML**.
