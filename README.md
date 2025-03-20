# Audio Deepfake Detection using Wav2Vec 2.0

## Overview
This project focuses on **detecting AI-generated human speech** using **Wav2Vec 2.0** fine-tuned on the **Fake-or-Real (FoR) Dataset**. The goal is to classify speech as either **real or deepfake** using self-supervised learning.

## Repository Structure
```
📂 Audio-Deepfake-Detection
├── 📂 data                  # Dataset (link and instructions for downloading)
├── 📂 notebooks             # Jupyter notebooks for data processing and training
├── 📂 models                # Saved trained models
├── 📂 scripts               # Python scripts for data processing and evaluation
├── 📜 README.md             # Project documentation
├── 📜 requirements.txt      # Required dependencies
└── 📜 train.py              # Training script
```

## Problem Statement
AI-generated speech can be used for **fraudulent activities**, **identity theft**, and **misinformation**. This project aims to build a **deepfake detection model** capable of real-time detection.

## Selected Models & Approaches
### 1️ Wav2Vec 2.0 + Transformer-based Classification (Implemented)
- **Key Innovation**: Learns speech features from raw waveforms using a Transformer classifier.
- **Performance**: 96%+ accuracy on ASVspoof 2019 dataset.
- **Why It’s Promising**: High accuracy, adaptable to evolving deepfake speech techniques.
- **Limitations**: Requires GPU acceleration for real-time detection.
- **Use Case**: Fraud detection, media verification.

### 2️⃣ RawNet2 – CNN-based End-to-End Deepfake Detection
- **Key Innovation**: Uses CNNs to process raw waveforms for deepfake detection.
- **Performance**: Equal Error Rate (EER) of 2.5% on ASVspoof 2019 dataset.
- **Why It’s Promising**: Fast inference, works well in noisy environments.
- **Limitations**: Prone to overfitting, black-box nature.
- **Use Case**: Real-time detection in phone calls, meetings.

### 3️⃣ Whisper-based Acoustic & Linguistic Anomaly Detection
- **Key Innovation**: Uses Whisper ASR + NLP to detect anomalies in speech and text.
- **Performance**: High accuracy even in noisy conditions.
- **Why It’s Promising**: Dual-layer detection enhances robustness.
- **Limitations**: High computational cost, requires GPU.
- **Use Case**: Forensic analysis, regulatory compliance.

## 📂 Dataset
- **Name**: The Fake-or-Real (FoR) Dataset (Deepfake Audio)
- **Source**: [Kaggle Link](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)
- **Structure**:
  - **Training**: `train/fake`, `train/real`
  - **Validation**: `validation/fake`, `validation/real`
  - **Testing**: `test/fake`, `test/real`

## Implementation Details
### 1️⃣ Data Preprocessing
- Convert `.wav` files to 16kHz using **Librosa**.
- Extract features using **Wav2Vec 2.0 Feature Extractor**.

### 2️⃣ Model Fine-Tuning
- **Pre-trained Model**: `facebook/wav2vec2-large-xlsr-53`
- **Hyperparameters**:
  - Batch size: `4`
  - Learning rate: `5e-5`
  - Optimizer: `AdamW`
  - Loss function: `CrossEntropyLoss`
  - Epochs: `3`

### 3️⃣ Evaluation
- **Metrics**: Accuracy, Confusion Matrix, Classification Report.
- **Results**:
  - Accuracy: **TBD after training**
  - Misclassification Rate: **TBD**

## Installation & Setup
### 1️⃣ Clone Repository
```bash
git clone https://github.com/yourusername/Audio-Deepfake-Detection.git
cd Audio-Deepfake-Detection
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Run Training
```bash
python train.py
```
### 4️⃣ Evaluate Model
```bash
python evaluate.py
```

## Challenges Faced
- **Computational Limitations**: Required **Colab GPU** for training.
- **Dataset Imbalance**: Addressed with **data augmentation**.
- **Model Overfitting**: Regularization techniques applied.

## 📝 Future Work
- Implement **RawNet2** & **Whisper** models for comparison.
- Deploy as a **real-time API** for fraud detection.
- Test on diverse **deepfake datasets** like ASVspoof 2021.

## 📜 References
- Baevski et al. (2020). Wav2Vec 2.0: Self-supervised learning of speech representations. [Link](https://doi.org/10.48550/arXiv.2006.11477)
- Tak et al. (2021). End-to-end deepfake speech detection using RawNet2. [Link](https://doi.org/10.1109/ICASSP39728.2021.9413977)
- Radford et al. (2022). Whisper: Robust speech recognition. [Link](https://cdn.openai.com/papers/whisper.pdf)
