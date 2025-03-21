#  AI-Generated Speech Forgery Detection

## Project Summary

With the increasing sophistication of AI-based text-to-speech (TTS) systems and voice cloning technologies, detecting forged or synthetic speech has become a critical challenge in fields like media verification, fraud prevention, law enforcement, and cybersecurity. This project explores cutting-edge deep learning models for **speech forgery detection**, focusing on:

- Evaluating three modern architectures
- Implementing and fine-tuning the **Wav2Vec 2.0 + Transformer** pipeline
- Benchmarking the model on a real-world dataset
- Analyzing gaps between research performance and practical outcomes

---

## Table of Contents

- [Model Overview](#model-overview)
- [Implementation Pipeline](#implementation-pipeline)
- [Experimental Results](#experimental-results)
- [Analysis and Discussion](#analysis-and-discussion)
- [Challenges & Future Directions](#challenges--future-directions)
- [References](#references)

---

##  Model Overview

### 1. Wav2Vec 2.0 + Transformer-Based Classifier

#### Technical Insight
- Developed by Facebook AI, **Wav2Vec 2.0** is a **self-supervised learning** framework that pre-trains models on **raw audio waveforms** without requiring manual feature extraction (like MFCC or spectrograms).
- Audio embeddings are passed through a **Transformer encoder** which captures contextual dependencies and subtle temporal features critical for forgery detection.

#### Research Performance
- Achieves over **96% accuracy** on benchmark datasets like **ASVspoof 2019**.
- Effective in identifying minute spectral inconsistencies in AI-generated speech.

####  Strengths
- Requires no manual feature engineering
- Generalizes well with transfer learning
- Adaptable to multiple languages and accents

####  Limitations
- Computationally heavy; needs GPU for inference
- Fine-tuning on domain-specific data required for best performance

#### Use Case
Ideal for **real-time or batch processing** in:
- Audio forensics
- Call center fraud detection
- Social media misinformation filtering

---

### 2. RawNet2 – End-to-End CNN-Based Detection

####  Technical Insight
- RawNet2 employs **1D convolutional layers** directly on waveform inputs.
- It uses **Residual and Squeeze-and-Excitation blocks** for effective feature extraction without needing spectrograms.

#### Research Performance
- Achieved **Equal Error Rate (EER) of 2.5%** on ASVspoof 2019.
- Known for fast inference and strong real-time performance.

####  Strengths
- Lightweight compared to Transformer models
- Works well under background noise
- Efficient for embedded/edge deployment

#### ⚠Limitations
- Overfits smaller datasets
- Lower interpretability than Transformer-based models

---

###  3. Whisper + Acoustic & Semantic Anomaly Detection

#### Technical Insight
- Whisper, an ASR model by OpenAI, offers robust transcription from raw audio.
- Combines **acoustic anomalies** (like pitch irregularities) and **linguistic patterns** (odd semantics, unnatural cadence) for dual-layer detection.

#### Research Performance
- State-of-the-art accuracy in noisy settings
- More resilient to **adversarial attacks** compared to waveform-only models

####  Strengths
- Best for **forensic transcription** and post-event analysis
- Can flag **synthetic inconsistencies** in word choice or delivery

#### Limitations
- Requires high compute for real-time transcription
- Not well-suited for ultra-short clips or low-quality samples

---

##  Implementation Pipeline

### Selected Model: Wav2Vec 2.0 + Transformer Classifier

### Environment
- Python 3.10
- `transformers`, `datasets`, `librosa`, `scikit-learn`, `torch`, `pandas`

### Steps

1. **Dataset Preparation**
   - Dataset: [Fake-or-Real (FoR)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/UV7NGU)
   - Labels: `REAL` (human speech) vs. `FAKE` (AI-generated speech)
   - All audio normalized and resampled to **16kHz**.

2. **Model & Tokenizer**
   - Pre-trained: `facebook/wav2vec2-large-xlsr-53`
   - Used **Hugging Face Trainer API** with customized head for binary classification.

3. **Fine-Tuning**
   - Epochs: 3
   - Optimizer: AdamW
   - Learning Rate: 2e-5
   - Batch size: 8 (limited by Colab GPU)

4. **Evaluation**
   - Metrics: Accuracy, Confusion Matrix, Precision, Recall, F1-score
   - Toolkits: `sklearn.metrics`, `matplotlib`

---

## Experimental Results

| Metric        | Value       |
|---------------|-------------|
| Accuracy      | 50%         |
| Precision     | 0.48        |
| Recall        | 0.52        |
| F1-score      | 0.49        |

### Confusion Matrix Analysis

|               | Predicted Real | Predicted Fake |
|---------------|----------------|----------------|
| **Actual Real** | 52             | 48             |
| **Actual Fake** | 50             | 50             |

#### Key Observations:
- The model misclassified 48% of real speech as fake, indicating high false positives.

- 50% of fake speech was incorrectly classified as real, highlighting false negatives.

- The model struggles with distinguishing deepfake voices, suggesting deeper spectral features may be required.
---

## Analysis and Discussion

### Why the Model Underperformed

1. **Dataset Imbalance**  
   - Real speech dominates training data, creating bias in predictions.

2. **Lack of Domain Adaptation**  
   - Research models were trained on ASVspoof; our model was fine-tuned on a **different domain** (FoR).

3. **Signal Similarity**  
   - Deepfake voices have become acoustically **indistinguishable** from human voices with current synthesis methods.

---

### Lessons Learned

- Benchmark success ≠ real-world success.
- Speech detection is **contextual**, and deepfake detection benefits from **multi-modal inputs** (audio + text).
- Need for a **richer dataset** including varied accents, environments, and forgery techniques.

---

### Challenges & Future Directions

## Implementation Challenges Encountered
During the development and training of the deepfake speech detection model, several challenges emerged:
# Computational Limitations: 
The training process relied on Google Colab GPUs, which posed constraints due to limited memory and frequent session timeouts.
# Dataset Imbalance: 
The Fake-or-Real dataset exhibited class imbalance. Data augmentation techniques were employed to address this issue.
# Model Overfitting: 
Overfitting was mitigated using dropout regularization and early stopping mechanisms during training.
# Generalization Issues:
The model demonstrated strong performance on the Fake-or-Real dataset, but exhibited high variance during validation, indicating limited generalization across other domains.
# Noisy Training Samples:
The presence of low-quality or noisy audio samples negatively affected the quality of learned feature embeddings.

### Future Enhancements
To improve the robustness and accuracy of the deepfake detection model, the following enhancements are proposed:
## Domain Adaptation:
Implement adversarial training techniques to improve performance across diverse speech domains.
## Ensemble Modeling:
Combine CNN, Transformer, and ASR-based models to capture different aspects of speech features.
## Custom Loss Functions:
Develop and apply loss functions tailored to reducing false positives, enhancing detection precision.
## Larger and More Diverse Datasets:
Expand the training dataset to include varied voices, accents, and audio conditions for better generalization.

---

##  References

1. **Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020).**  
   *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.*  
   arXiv: [2006.11477](https://arxiv.org/abs/2006.11477)

2. **Tak, H., Patwa, P., & Yamagishi, J. (2021).**  
   *End-to-End Detection of Spoofed Speech Using RawNet2.*  
   In *ICASSP 2021 - IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*.  
   DOI: [10.1109/ICASSP39728.2021.9413753](https://ieeexplore.ieee.org/document/9413753)

3. **Radford, A., et al. (2022).**  
   *Whisper: Robust Speech Recognition via Large-Scale Weak Supervision.*  
   OpenAI. [https://openai.com/research/whisper](https://openai.com/research/whisper)

4. **ASVspoof Challenge Dataset.**  
   [https://www.asvspoof.org](https://www.asvspoof.org)

5. **Fake-or-Real Speech Dataset.**  
      - Kaggle Mirror: [https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)

6. **Hugging Face Wav2Vec2.0 Model:**  
   [https://huggingface.co/facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53)
---
## Acknowledgments

- Thanks to Hugging Face, Google Colab, OpenAI, and Harvard Dataverse for open access to incredible tools and datasets.
- Special credit to the research community for pioneering this domain and making it publicly accessible.

---

##  Repository Structure

```plaintext
📦 AI-Speech-Forgery-Detection/
 ┣ 📁 data/                   # Processed audio files and labels
 ┣ 📁 models/                 # Saved model checkpoints
 ┣ 📁 notebooks/              # Jupyter Notebooks for experiments and visualizations
 ┣ 📁 utils/                  # Helper functions (e.g., data loaders, evaluation scripts)
 ┣ 📄 README.md               # Project documentation
 ┣ 📄 requirements.txt        # Python dependencies
 ┣ 📄 train.py                # Main training script
 ┣ 📄 evaluate.py             # Script to evaluate model performance
 ┗ 📄 inference.py            # Script for real-time prediction or batch inference

---

##  Setup Instructions

```bash
git clone https://github.com/<your-username>/AI-Speech-Forgery-Detection.git
cd AI-Speech-Forgery-Detection
pip install -r requirements.txt


