# MACHINE LEARNING COURSE REPORT
# Topic: Heart Sound Classification Using CNN-LSTM on Mel Spectrogram

---

## 1. Introduction

### 1.1. Problem Statement

Cardiovascular disease is the leading cause of death worldwide, accounting for approximately 17.9 million deaths per year (WHO, 2023). Early detection of abnormalities in heart sounds plays a crucial role in screening and diagnosing cardiovascular conditions.

Traditional cardiac auscultation depends on the experience and skill of physicians, leading to inter-observer variability. Therefore, applying **Machine Learning** to automatically classify heart sounds is a highly promising approach.

### 1.2. Objectives

- Build a **CNN-LSTM** model to classify heart sounds into 2 classes: **Normal** and **Abnormal**
- Achieve accuracy **≥ 90%** on the test set
- Deploy a mobile application that allows users to upload audio recordings and receive diagnostic results

### 1.3. Scope

- **Data**: PhysioNet/CinC Challenge 2016 dataset
- **Architecture**: CNN-LSTM Hybrid on Log-Mel Spectrogram
- **Deployment**: FastAPI API + React Native (Expo) mobile application

---

## 2. Theoretical Background

### 2.1. Heart Sounds

A normal cardiac cycle consists of two main sounds:
- **S1 (First Heart Sound)**: Produced when the mitral and tricuspid valves close, marking the beginning of systole. Frequency: 10-140 Hz.
- **S2 (Second Heart Sound)**: Produced when the aortic and pulmonary valves close, marking the beginning of diastole. Frequency: 10-400 Hz.

Abnormal heart sounds (murmurs) occur due to turbulent blood flow through damaged valves or structural heart abnormalities.

### 2.2. Mel Spectrogram

The Mel Spectrogram is a time-frequency representation of an audio signal, where the frequency axis is converted to the Mel scale — simulating how the human ear perceives frequency:

$$m = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$$

**Log-Mel Spectrogram** is computed through the following steps:
1. Divide the signal into frames using a Hanning window
2. Compute the FFT (Fast Fourier Transform) for each frame
3. Apply a Mel filterbank to the power spectrum
4. Convert to logarithmic scale (dB)

### 2.3. Delta Features

To capture information about the **rate of change** of the spectrum over time, we use delta features:

- **Delta (Δ)** — First-order difference: Represents the rate of energy change at each frequency bin over time
- **Delta-Delta (Δ²)** — Second-order difference: Represents the acceleration of change, helping detect sudden transitions

The final input to the model is a 3-channel tensor:
```
[Log-Mel Spectrogram | Delta | Delta-Delta] → Shape: [64, T, 3]
```

### 2.4. Convolutional Neural Network (CNN)

CNNs work effectively on spatially structured data (images, spectrograms) through:
- **Convolutional layers**: Extract local features (frequency patterns)
- **Pooling layers**: Reduce dimensionality, increase invariance
- **Batch Normalization**: Stabilize the training process

### 2.5. Long Short-Term Memory (LSTM)

LSTM addresses the vanishing gradient problem in RNNs, enabling learning of long-term dependencies. In this task, LSTM learns:
- The repeating rhythm S1 → S2 → S1 → S2
- Time intervals between heartbeats
- Abnormal patterns appearing in temporal sequences

---

## 3. Data

### 3.1. PhysioNet/CinC Challenge 2016 Dataset

| Information | Details |
|---|---|
| Source | PhysioNet/Computing in Cardiology Challenge 2016 |
| Number of recordings | ~3,240 recordings from 764 patients |
| Recording device | Medical-grade digital stethoscope |
| Auscultation sites | 4 standard positions: mitral, tricuspid, aortic, pulmonary |
| Labels | Normal, Abnormal (classified by cardiology specialists) |
| Original sample rate | 2000 Hz |

### 3.2. Data Preprocessing

#### Step 1: Resampling
The original signal is resampled to **SR = 4000 Hz** for standardization.

#### Step 2: Noise Reduction — Butterworth Bandpass Filter
A **4th-order Butterworth bandpass filter** with passband **25-900 Hz** is applied:

```python
BP_LOW = 25.0   # Hz — removes low-frequency noise (movement, respiration)
BP_HIGH = 900.0 # Hz — removes high-frequency noise (electrical, environmental)
BP_ORDER = 4    # Filter order — balance between roll-off steepness and stability
```

Transfer function of the Nth-order Butterworth filter:

$$|H(j\omega)|^2 = \frac{1}{1 + \left(\frac{\omega}{\omega_c}\right)^{2N}}$$

#### Step 3: Segmentation
Each recording is divided into **5-second segments** with a hop of **2.5 seconds** (50% overlap):

| Parameter | Value |
|---|---|
| Segment length | 5.0 seconds (20,000 samples) |
| Hop size | 2.5 seconds (50% overlap) |
| Padding | Zero-padding if segment is shorter than 5 seconds |

#### Step 4: Feature Extraction — Log-Mel Spectrogram + Delta

| Parameter | Value | Description |
|---|---|---|
| N_FFT | 512 | FFT window size |
| HOP_LENGTH | 128 | FFT hop length |
| N_MELS | 64 | Number of Mel filters |
| FMIN | 20 Hz | Lowest frequency |
| FMAX | 2000 Hz | Highest frequency |

Output: Tensor **[64, 157, 3]** per segment (64 mel bins × 157 time frames × 3 channels).

### 3.3. Data Splitting

| Set | Number of segments | Ratio |
|---|---|---|
| Train | ~17,064 | 70% |
| Validation | ~3,652 | 15% |
| Test | ~3,652 | 15% |

Split by **record_id** (not by segment) to prevent data leakage.

### 3.4. Class Balancing (Class Weighting)

Due to class imbalance (Normal >> Abnormal), we use automatic **class weights**:

```python
class_weight = {
    idx: total / (num_classes * count)
    for idx, count in class_counts.items()
}
```

---

## 4. Model Architecture

### 4.1. CNN-LSTM Hybrid Overview

```
Input [64, T, 3]
    │
    ├── CNN Feature Extractor
    │   ├── Conv2D(32, 3×3) + BN + ReLU + MaxPool(2×2)
    │   ├── Conv2D(64, 3×3) + BN + ReLU + MaxPool(2×2)
    │   └── Conv2D(128, 3×3) + BN + ReLU + MaxPool(2×1) + Dropout(0.3)
    │
    ├── Reshape → [T', features]
    │   └── Permute + TimeDistributed(Flatten)
    │
    ├── LSTM Temporal Learner
    │   └── Bidirectional LSTM (64 units)
    │   └── Dropout(0.3)
    │
    └── Classification Head
        ├── Dense(64, ReLU) + Dropout(0.15)
        └── Dense(2, Softmax) → [Normal, Abnormal]
```

### 4.2. Component Details

#### CNN Feature Extractor (3 blocks)

| Block | Filters | Kernel | Pooling | Output shape |
|---|---|---|---|---|
| Block 1 | 32 | 3×3 | 2×2 | [32, T/2, 32] |
| Block 2 | 64 | 3×3 | 2×2 | [16, T/4, 64] |
| Block 3 | 128 | 3×3 | **2×1** | [8, T/4, 128] |

> **Note**: Block 3 uses MaxPool(2,1) — pooling only along the frequency axis, preserving temporal resolution for better LSTM processing.

#### LSTM Temporal Learner

- **Bidirectional LSTM** with 64 units (→ 128 output features)
- `return_sequences=False`: Only takes the final output
- Learns heartbeat cycle patterns: S1→pause→S2→pause→S1

#### Classification Head

- Dense(64) + ReLU: Dimensionality reduction
- Dense(2) + Softmax: Probability [Normal, Abnormal]

### 4.3. Regularization Techniques

| Technique | Details |
|---|---|
| Dropout | 0.3 after final CNN block and LSTM; 0.15 before output |
| Batch Normalization | After each Conv2D layer |
| SpecAugment | Random masking on frequency and time axes (p=0.5) |
| Early Stopping | patience=6, monitoring val_accuracy |
| Class Weighting | Automatic minority class balancing |

---

## 5. Training

### 5.1. Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 (with ReduceLROnPlateau, factor=0.5) |
| Batch size | 32 |
| Max epochs | 30 |
| Loss function | Sparse Categorical Cross-Entropy |
| Early stopping | patience=6 epochs, monitor=val_accuracy |
| Random seed | 42 (reproducibility) |

### 5.2. Data Augmentation — SpecAugment

SpecAugment is applied on the training set with 50% probability for each mask type:

- **Frequency masking**: Randomly masks up to 8 consecutive frequency bins
- **Time masking**: Randomly masks up to 16 consecutive time frames

Purpose: Improve generalization, reduce overfitting, simulate real-world noise.

### 5.3. Training Progress

| Epoch | Train Accuracy | Train Loss |
|---|---|---|
| 1 | 72.1% | 0.536 |
| 5 | 86.9% | 0.228 |
| 10 | 92.1% | 0.146 |
| 15 | 91.6% | 0.160 |
| 20 | 93.1% | 0.131 |
| 25 | 93.8% | 0.115 |
| 30 | 94.8% | 0.105 |
| **Best** | **95.0%** | **0.101** |

Training converged steadily with accuracy increasing across epochs.

---

## 6. Evaluation Results

### 6.1. Test Set Results

**Weighted F1-Score: 0.936 (93.6%)**

#### Classification Report (per-record):

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| abnormal_other | 0.87 | 0.84 | 0.86 | ~100 |
| normal | 0.96 | 0.97 | 0.96 | ~340 |
| **weighted avg** | **0.94** | **0.94** | **0.936** | **~440** |

### 6.2. Confusion Matrix

#### Confusion Matrix (Records)

The model correctly classifies the majority of cases:
- **True Normal → Predicted Normal**: High rate (~97%)
- **True Abnormal → Predicted Abnormal**: Good rate (~84%)
- **False Positive** (Normal misclassified as Abnormal): Low (~3%)
- **False Negative** (Abnormal misclassified as Normal): Needs improvement (~16%)

### 6.3. ROC Curve

The ROC curve illustrates the model's ability to distinguish between the two classes. AUC (Area Under Curve) closer to 1 indicates better performance.

### 6.4. Comparison with Related Work

| Study | Architecture | Dataset | F1-Score |
|---|---|---|---|
| Deng et al. (2020) | CRNN + MFCC | PhysioNet 2016 | 90.06% |
| **IConNet (MobiSys 2024)** | Lightweight CNN | PhysioNet 2016 | 92.05% |
| Li et al. (2022) | ResNet + MFCC | PhysioNet 2016 | ~94% |
| **Our model** | **CNN-LSTM + Log-Mel + Delta** | **PhysioNet 2016** | **93.6%** |

→ Our results outperform IConNet (2024) and are comparable to Li et al. (2022).

---

## 7. Application Deployment

### 7.1. System Architecture

```
┌──────────────┐     HTTPS/Tunnel     ┌──────────────────┐
│  Mobile App  │  ◄──────────────────► │   FastAPI Server │
│  React Native│     JSON + Base64     │   + TensorFlow   │
│  (Expo)      │                       │   + Librosa      │
└──────────────┘                       └──────────────────┘
     │                                        │
     │ Upload WAV/MP3                         │ Load model
     │ Receive results                        │ Predict
     │ Display BPM                            │ Gen spectrogram
     └────────────────────────────────────────┘
```

### 7.2. API Endpoint

```
POST /api/predict
Content-Type: multipart/form-data
Body: file=<audio_file>

Response (JSON):
{
    "primary_prediction": "normal" | "abnormal_other",
    "confidence": 0.996,
    "bpm": 88,
    "signal_quality": 1.0,
    "recommendation": "Normal heartbeat...",
    "spectrogram_b64": "data:image/png;base64,..."
}
```

### 7.3. Mobile Application Interface

The application displays:
- **Diagnostic status**: Normal (green) / Abnormal (red)
- **Accuracy**: Model confidence score
- **BPM**: Heart rate estimated using autocorrelation
- **Signal quality**: Quality of the recorded signal
- **Frequency spectrum**: Log-Mel Spectrogram image
- **Recommendation**: Appropriate medical advice

---

## 8. Discussion

### 8.1. Strengths

1. **Hybrid CNN-LSTM architecture** combines the advantages of both:
   - CNN extracts frequency patterns from the spectrogram
   - LSTM learns temporal rhythms of heartbeats
2. **3-channel input** (Static + Delta + Delta²) provides richer information than single-channel
3. **SpecAugment** effectively reduces overfitting
4. **Class weighting** handles data imbalance well

### 8.2. Limitations

1. **Domain mismatch**: Model trained on medical stethoscope data, not optimized for smartphone microphones
2. **False Negatives**: ~16% of abnormal cases misclassified as normal — needs improvement for medical applications
3. **Limited data**: Only uses 1 dataset (PhysioNet 2016)

### 8.3. Future Work

1. **Phone-specific data collection**: Fine-tune model on smartphone microphone recordings
2. **Stethoscope adapter**: Use low-cost accessories to improve recording quality
3. **On-device inference**: Convert model to TensorFlow Lite for direct on-phone execution
4. **Multi-class classification**: Extend to classify specific types of heart diseases

---

## 9. Conclusion

This study successfully built a heart sound classification system using a **CNN-LSTM Hybrid** model on **Log-Mel Spectrogram** combined with **Delta/Delta-Delta features**. The system achieved:

- **Training accuracy**: 95.0%
- **Weighted F1-score**: 93.6% (on test set)
- **Outperformed** IConNet (MobiSys 2024, F1=92.05%)

The system has been successfully deployed on a mobile application, allowing users to upload audio recordings and receive diagnostic results with comprehensive information: BPM, signal quality, spectrogram visualization, and medical recommendations.

---

## 10. References

1. Clifford, G.D., et al. (2016). "Classification of normal/abnormal heart sound recordings: the PhysioNet/Computing in Cardiology Challenge 2016." *Computing in Cardiology*, 43, 609-612.

2. Vu, L., & Tran, T. (2024). "Detecting abnormal heart sound using mobile phones and on-device IConNet." *arXiv:2412.03267*. MobiSys 2024.

3. Makimoto, H., et al. (2022). "Efficient screening for severe aortic valve stenosis using understandable artificial intelligence." *European Heart Journal – Digital Health*, 3(2), 165-173.

4. Deng, M., Meng, T., et al. (2020). "Heart sound classification based on improved MFCC features and convolutional recurrent neural networks." *Neural Networks*, 130, 22-32.

5. Li, F., Zhang, Z., Wang, L., & Liu, W. (2022). "Heart sound classification based on improved mel-frequency spectral coefficients and deep residual learning." *Frontiers in Physiology*, 13, 1084420.

6. Park, D.S., et al. (2019). "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition." *Proc. Interspeech 2019*.

---

## Appendix

### A. Project Structure

```
heart-sound-classification/
├── data/
│   ├── raw/physionet_2016/     # Original dataset
│   ├── processed/logmels/      # Extracted features
│   └── metadata/               # Label maps, splits, norm stats
├── models/tf_heart_sound/      # Trained model
├── src/
│   ├── preprocess/
│   │   └── extract_features.py # Log-Mel + Delta extraction
│   ├── training/
│   │   ├── model_cnn_lstm.py   # CNN-LSTM architecture
│   │   └── train_tf.py         # Training script
│   ├── inference/
│   │   ├── audio_utils.py      # Inference preprocessing
│   │   └── predict_tf.py       # Prediction + BPM + Signal Quality
│   └── app/
│       └── main.py             # FastAPI server
├── reports/figures/            # Confusion matrices, ROC curve
└── heart-sound-app/            # React Native mobile app
```

### B. Experimental Environment

| Component | Details |
|---|---|
| Language | Python 3.11 |
| ML Framework | TensorFlow 2.16 / Keras 3 |
| Audio Processing | Librosa, SciPy |
| API Server | FastAPI + Uvicorn |
| Mobile App | React Native (Expo) |
| Operating System | Windows 11 |
