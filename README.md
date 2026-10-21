# 👗 Fashion-MNIST Classification: Comparing Deep Learning Architectures

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19%2B-orange.svg)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end deep learning benchmark evaluating the effect of model complexity on the **Fashion-MNIST** dataset. This project contrasts a standard Multi-Layer Perceptron (**ANN**) with a streamlined **Basic CNN** and a regularized **Deeper CNN** (featuring Batch Normalization and Dropout).

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset Details](#-dataset-details)
- [Model Architectures](#-model-architectures)
- [Training & Regularization Strategy](#-training--regularization-strategy)
- [Performance & Benchmark Results](#-performance--benchmark-results)
- [Qualitative Prediction Analysis](#-qualitative-prediction-analysis)
- [Key Insights & Findings](#-key-insights--findings)
- [Installation & Quickstart](#-installation--quickstart)

---

## 📖 Project Overview

Fashion-MNIST serves as a drop-in replacement for classical MNIST digits, featuring grayscale product images from 10 clothing categories. The goal of this experiment is to explore:
1. The performance boost attained when moving from flat feed-forward layers to spatial convolutional kernels.
2. The trade-offs between network depth, regularization (Batch Normalization & Dropout), compute footprint, and generalization on low-resolution ($28 \times 28$) imagery.

---

## 📊 Dataset Details

- **Dataset**: Zalando's [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
- **Samples**: 60,000 training images, 10,000 test images
- **Dimensions**: $28 \times 28 \times 1$ (single-channel grayscale)
- **Target Classes (10 categories)**:
  `0: T-shirt/top`, `1: Trouser`, `2: Pullover`, `3: Dress`, `4: Coat`, `5: Sandal`, `6: Shirt`, `7: Sneaker`, `8: Bag`, `9: Ankle boot`

### Data Preprocessing Pipeline
1. **Min-Max Normalization**: Image arrays scaled from $[0, 255]$ integer intensities to float values in the range $[0.0, 1.0]$.
2. **Channel Expansion**: Grayscale arrays reshaped from `(N, 28, 28)` to `(N, 28, 28, 1)` for 2D convolution compatibility.
3. **One-Hot Encoding**: Integer class indices mapped to binary vectors of length 10 using `keras.utils.to_categorical`.

---

## 🧠 Model Architectures

### 1. Basic ANN (Multilayer Perceptron)
- **Input**: Flatten $(28 \times 28 \rightarrow 784)$
- **Hidden Layer 1**: Dense ($128$ units, ReLU)
- **Hidden Layer 2**: Dense ($64$ units, ReLU)
- **Output Layer**: Dense ($10$ units, Softmax)
- **Parameters**: **109,386** (All trainable)

### 2. Basic CNN
- **Conv Block 1**: Conv2D ($32$ filters, $3 \times 3$, ReLU) $\rightarrow$ MaxPooling2D ($2 \times 2$)
- **Conv Block 2**: Conv2D ($64$ filters, $3 \times 3$, ReLU) $\rightarrow$ MaxPooling2D ($2 \times 2$)
- **Classifier**: Flatten ($1,600$ units) $\rightarrow$ Dense ($64$ units, ReLU) $\rightarrow$ Dense ($10$ units, Softmax)
- **Parameters**: **121,930** (All trainable)

### 3. Deeper Regularized CNN
- **Conv Block 1**: Conv2D ($32$, $3 \times 3$, ReLU) $\rightarrow$ BatchNorm $\rightarrow$ MaxPool ($2 \times 2$) $\rightarrow$ Dropout ($0.25$)
- **Conv Block 2**: Conv2D ($64$, $3 \times 3$, ReLU) $\rightarrow$ BatchNorm $\rightarrow$ MaxPool ($2 \times 2$) $\rightarrow$ Dropout ($0.25$)
- **Conv Block 3**: Conv2D ($128$, $3 \times 3$, ReLU) $\rightarrow$ BatchNorm $\rightarrow$ MaxPool ($2 \times 2$) $\rightarrow$ Dropout ($0.25$)
- **Classifier**: Flatten ($128$ units) $\rightarrow$ Dense ($128$ units, ReLU) $\rightarrow$ BatchNorm $\rightarrow$ Dropout ($0.50$) $\rightarrow$ Dense ($10$ units, Softmax)
- **Parameters**: **111,882** (Trainable: 111,178 | Non-trainable: 704)

---

## ⚙️ Training & Regularization Strategy

- **Optimizer**: Adam ($\eta = 0.001$)
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 64
- **Max Epochs**: 30
- **Callbacks**:
  - `EarlyStopping`: Monitored `val_loss` with a patience of 5 epochs; automatically restored the best weights.
  - `ModelCheckpoint`: Stored peak-performing weights per model (`best_ann_model_weights.weights.h5`, `best_basic_cnn_model_weights.weights.h5`, `best_deeper_cnn_model_weights.weights.h5`).

---

## 📈 Performance & Benchmark Results

All evaluations were executed on the held-out test split ($10,000$ unseen images) after restoring the checkpointed optimal weights:

| Model Architecture | Parameters | Stopping Epoch | Test Loss | Test Accuracy | Generalization Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **ANN (Dense)** | 109,386 | 18 | ~0.3245 | ~88.79% | Suffered from early loss plateaus; lacks spatial translation invariance. |
| **Basic CNN** 🏆 | 121,930 | 12 | **~0.2483** | **~91.30%** | **Fastest convergence & lowest test loss.** Optimal balance for $28 \times 28$ resolution. |
| **Deeper CNN** | 111,882 | 30 | ~0.2645 | ~90.57% | Effective regularization (Dropout/BN), but 3 pooling steps reduced features down to $1 \times 1$ spatial grids. |

### Diagnostic Metrics Included:
- **Confusion Matrices**: Assessed inter-class confusion (common misclassifications between visually similar apparel such as *Shirt* vs. *T-shirt/top* and *Coat* vs. *Pullover*).
- **Interactive Plotly Visualizations**: Plotted dual training & validation loss and accuracy trajectories side-by-side.

---

## 🔍 Qualitative Prediction Analysis

Using the best-performing model (**Basic CNN**):
- A randomly sampled, shuffled subset of 200 items (100 accurate, 100 misclassified) was visualized in a $20 \times 10$ subplot grid.
- **Accurate** labels displayed in **green**; **misclassifications** highlighted in **red** with predicted versus ground-truth targets ($P:\dots / T:\dots$).
- Primary error patterns occurred along subtle texture/collar boundaries:
  - `Shirt` categorized as `T-shirt/top` or `Coat`
  - `Sneaker` vs. `Ankle boot` ambiguity on low-profile footwear silhouettes.

---

## 💡 Key Insights & Findings

1. **Convolutions Outperform Flat MLPs**: Transitioning from flattened dense layers to basic 2D convolutions yielded an immediate **~2.5% accuracy gain** and reduced test loss from `0.32` to `0.24`.
2. **The Perils of Over-Pooling Small Inputs**: For $28 \times 28$ images, three consecutive $2 \times 2$ max-pooling operations downsample feature map spatial dimensions drastically ($28 \rightarrow 13 \rightarrow 5 \rightarrow 1$). The final convolution operated on a spatial size of $1 \times 1$, leading to spatial information bottlenecking in the Deeper CNN.
3. **Efficiency vs. Complexity**: The **Basic CNN** converged in only **12 epochs**, providing the optimal trade-off between training speed, parameter count, and validation accuracy.

---

## 🚀 Installation & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/fashion-mnist-classification.git
cd fashion-mnist-classification
```

### 2. Configure Environment
It is recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Dependencies (`requirements.txt`)
```text
tensorflow>=2.19.0
keras>=3.5.0
numpy>=1.26.0
pandas>=2.0.0
matplotlib>=3.8.0
seaborn>=0.13.0
scikit-learn>=1.4.0
plotly>=5.18.0
```

### 4. Run the Notebook
Launch via Jupyter Lab or Google Colab:
```bash
jupyter notebook fashion_mnist_classification.ipynb
```
