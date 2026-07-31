# Quantum Computing

This repository contains different projects, benchmarks and experiments related to **Quantum Computing**, **Quantum Machine Learning (QML)** and quantum software frameworks.

## Repository structure

### 📚 Bibliography

Collection of books, scientific papers and reference material related to:

- Quantum Computing
- Quantum Machine Learning
- Quantum Algorithms
- Quantum Information
- Quantum Software Frameworks

This folder serves as the theoretical foundation for the rest of the projects.

---

### 🖼️ QED (Quantum Edge Detection)

Implementation and experiments on **Quantum Edge Detection (QED)** for image processing.

The project explores quantum image representations and quantum circuits for edge detection, comparing their behaviour with classical approaches.

---

### ⚡ benchmarks_emuladores

Benchmark suite for comparing different **quantum simulators/emulators** using Shor's algorithm.

Frameworks evaluated include:

- Qiskit
- Qulacs
- myQLM

The objective is to analyse execution time, scalability and simulator performance.

---

### 🔄 benchmarks_QFT

Performance benchmark of the **Quantum Fourier Transform (QFT)** implemented using different quantum software frameworks.

Implemented with:

- Qiskit
- Qibo
- PennyLane
- Qulacs

The benchmarks compare execution time and scalability across frameworks.

---

### 🐑 lamb_QML

Quantum Machine Learning project for **lamb weight prediction** from image-derived morphological features.

The project includes both **classification** and **regression** approaches using classical and hybrid quantum machine learning models.

#### Structure

```
lamb_QML/
│
├── src/
│   ├── classification/
│   └── regression/
│
└── results/
    ├── classification/
    └── regression/
```

#### Classification

Comparison between:

- Classical SVC (Linear)
- Classical SVC (RBF)
- Quantum Support Vector Classifier (QSVC)

Evaluation metrics include:

- Accuracy
- Confusion Matrix

#### Regression

Comparison between:

- Classical SVR
- Quantum Kernel SVR

Evaluation metrics include:

- MAE
- RMSE
- R² score

The objective is to evaluate the capability of hybrid quantum machine learning methods for estimating lamb weight and compare their performance with classical machine learning models.

---

## Technologies

- Python
- Qiskit
- Qiskit Machine Learning
- PennyLane
- Qibo
- Qulacs
- myQLM
- Scikit-learn
- NumPy
- Pandas
- Matplotlib

---

## Author

Repository maintained for research and development in **Quantum Computing** and **Quantum Machine Learning**.
