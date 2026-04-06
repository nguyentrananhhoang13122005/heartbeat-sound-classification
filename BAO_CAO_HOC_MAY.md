# BÁO CÁO MÔN HỌC MÁY CƠ BẢN
# Đề tài: Phân loại âm thanh tim bằng mạng CNN-LSTM trên phổ Mel

---

## 1. Giới thiệu

### 1.1. Đặt vấn đề

Bệnh tim mạch là nguyên nhân tử vong hàng đầu trên toàn thế giới, chiếm khoảng 17.9 triệu ca tử vong mỗi năm (WHO, 2023). Việc phát hiện sớm các bất thường trong âm thanh tim đóng vai trò quan trọng trong sàng lọc và chẩn đoán bệnh tim mạch.

Phương pháp nghe tim truyền thống (auscultation) phụ thuộc vào kinh nghiệm và kỹ năng của bác sĩ, dẫn đến sai số giữa các người đánh giá (inter-observer variability). Do đó, việc ứng dụng **học máy (Machine Learning)** để tự động phân loại âm thanh tim là một hướng tiếp cận có tiềm năng lớn.

### 1.2. Mục tiêu

- Xây dựng mô hình **CNN-LSTM** để phân loại âm thanh tim thành 2 lớp: **Bình thường (Normal)** và **Bất thường (Abnormal)**
- Đạt độ chính xác **≥ 90%** trên tập kiểm tra
- Triển khai ứng dụng di động cho phép người dùng tải file thu âm và nhận kết quả chẩn đoán

### 1.3. Phạm vi

- **Dữ liệu**: Bộ dữ liệu PhysioNet/CinC Challenge 2016
- **Kiến trúc**: CNN-LSTM Hybrid trên Log-Mel Spectrogram
- **Triển khai**: API FastAPI + Ứng dụng di động React Native (Expo)

---

## 2. Cơ sở lý thuyết

### 2.1. Âm thanh tim

Một chu kỳ tim bình thường bao gồm hai âm thanh chính:
- **S1 (tiếng tim thứ nhất)**: Phát sinh khi van hai lá và van ba lá đóng, đánh dấu bắt đầu tâm thu. Tần số: 10-140 Hz.
- **S2 (tiếng tim thứ hai)**: Phát sinh khi van động mạch chủ và van phổi đóng, đánh dấu bắt đầu tâm trương. Tần số: 10-400 Hz.

Các âm thanh bất thường (murmurs) xuất hiện do dòng máu chảy rối (turbulent flow) qua các van bị tổn thương hoặc các bất thường cấu trúc tim.

### 2.2. Phổ Mel (Mel Spectrogram)

Phổ Mel là biểu diễn tần số-thời gian của tín hiệu âm thanh, trong đó trục tần số được chuyển đổi sang thang Mel — mô phỏng cách tai người cảm nhận tần số:

$$m = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$$

**Log-Mel Spectrogram** được tính qua các bước:
1. Chia tín hiệu thành các khung (frame) bằng cửa sổ Hanning
2. Tính FFT (Fast Fourier Transform) cho mỗi khung
3. Áp dụng bộ lọc Mel (mel filterbank) lên phổ công suất
4. Chuyển sang thang logarithm (dB)

### 2.3. Đặc trưng sai phân (Delta Features)

Để nắm bắt thông tin về **tốc độ biến đổi** của phổ theo thời gian, chúng tôi sử dụng đặc trưng sai phân:

- **Delta (Δ)** — Sai phân bậc 1: Thể hiện tốc độ thay đổi năng lượng tại mỗi bin tần số theo thời gian
- **Delta-Delta (Δ²)** — Sai phân bậc 2: Thể hiện gia tốc thay đổi, giúp phát hiện các chuyển tiếp đột ngột

Input cuối cùng vào mô hình là tensor 3 kênh:
```
[Log-Mel Spectrogram | Delta | Delta-Delta] → Shape: [64, T, 3]
```

### 2.4. Mạng Neural tích chập (CNN)

CNN hoạt động hiệu quả trên dữ liệu có cấu trúc không gian (ảnh, spectrogram) nhờ:
- **Convolutional layers**: Trích xuất đặc trưng cục bộ (pattern tần số)
- **Pooling layers**: Giảm chiều, tăng tính bất biến
- **Batch Normalization**: Ổn định quá trình huấn luyện

### 2.5. Mạng LSTM (Long Short-Term Memory)

LSTM giải quyết vấn đề vanishing gradient của RNN, cho phép học các phụ thuộc dài hạn. Trong bài toán này, LSTM học:
- Nhịp điệu lặp lại S1 → S2 → S1 → S2
- Khoảng cách thời gian giữa các nhịp tim
- Các pattern bất thường xuất hiện theo chuỗi thời gian

---

## 3. Dữ liệu

### 3.1. Bộ dữ liệu PhysioNet/CinC Challenge 2016

| Thông tin | Chi tiết |
|---|---|
| Nguồn | PhysioNet/Computing in Cardiology Challenge 2016 |
| Số lượng bản ghi | ~3,240 bản ghi từ 764 bệnh nhân |
| Thiết bị thu | Ống nghe điện tử y tế (digital stethoscope) |
| Vị trí đo | 4 vị trí chuẩn: van hai lá, van ba lá, van động mạch chủ, van phổi |
| Nhãn | Normal, Abnormal (được phân loại bởi bác sĩ chuyên khoa tim mạch) |
| Sample rate gốc | 2000 Hz |

### 3.2. Tiền xử lý dữ liệu

#### Bước 1: Resampling
Tín hiệu gốc được resample về **SR = 4000 Hz** để thống nhất.

#### Bước 2: Khử nhiễu — Lọc thông dải Butterworth
Áp dụng bộ lọc **Butterworth bậc 4** với dải thông **25-900 Hz**:

```python
BP_LOW = 25.0   # Hz — loại bỏ nhiễu tần số thấp (chuyển động, hô hấp)
BP_HIGH = 900.0 # Hz — loại bỏ nhiễu tần số cao (điện, môi trường)
BP_ORDER = 4    # Bậc lọc — cân bằng giữa độ dốc và ổn định
```

Hàm truyền của bộ lọc Butterworth bậc N:

$$|H(j\omega)|^2 = \frac{1}{1 + \left(\frac{\omega}{\omega_c}\right)^{2N}}$$

#### Bước 3: Phân đoạn (Segmentation)
Mỗi bản ghi được chia thành các **segment 5 giây** với bước trượt **2.5 giây** (overlap 50%):

| Tham số | Giá trị |
|---|---|
| Độ dài segment | 5.0 giây (20,000 samples) |
| Bước trượt | 2.5 giây (overlap 50%) |
| Padding | Zero-padding nếu segment ngắn hơn 5 giây |

#### Bước 4: Trích xuất đặc trưng Log-Mel Spectrogram + Delta

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| N_FFT | 512 | Kích thước cửa sổ FFT |
| HOP_LENGTH | 128 | Bước nhảy FFT |
| N_MELS | 64 | Số bộ lọc Mel |
| FMIN | 20 Hz | Tần số thấp nhất |
| FMAX | 2000 Hz | Tần số cao nhất |

Output: Tensor **[64, 157, 3]** cho mỗi segment (64 mel bins × 157 time frames × 3 channels).

### 3.3. Phân chia dữ liệu

| Tập | Số lượng segment | Tỉ lệ |
|---|---|---|
| Train | ~17,064 | 70% |
| Validation | ~3,652 | 15% |
| Test | ~3,652 | 15% |

Phân chia theo **record_id** (không phải segment) để tránh rò rỉ dữ liệu.

### 3.4. Cân bằng lớp (Class Weighting)

Do dữ liệu mất cân bằng (Normal >> Abnormal), chúng tôi sử dụng **class weights** tự động:

```python
class_weight = {
    idx: total / (num_classes * count)
    for idx, count in class_counts.items()
}
```

---

## 4. Kiến trúc mô hình

### 4.1. Tổng quan CNN-LSTM Hybrid

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

### 4.2. Chi tiết từng thành phần

#### CNN Feature Extractor (3 block)

| Block | Filters | Kernel | Pooling | Output shape |
|---|---|---|---|---|
| Block 1 | 32 | 3×3 | 2×2 | [32, T/2, 32] |
| Block 2 | 64 | 3×3 | 2×2 | [16, T/4, 64] |
| Block 3 | 128 | 3×3 | **2×1** | [8, T/4, 128] |

> **Lưu ý**: Block 3 sử dụng MaxPool(2,1) — chỉ pool theo trục tần số, giữ nguyên độ phân giải thời gian để LSTM xử lý tốt hơn.

#### LSTM Temporal Learner

- **Bidirectional LSTM** với 64 units (→ 128 output features)
- `return_sequences=False`: Chỉ lấy output cuối cùng
- Học pattern: S1→pause→S2→pause→S1 (chu kỳ nhịp tim)

#### Classification Head

- Dense(64) + ReLU: Giảm chiều
- Dense(2) + Softmax: Xác suất [Normal, Abnormal]

### 4.3. Các kỹ thuật regularization

| Kỹ thuật | Chi tiết |
|---|---|
| Dropout | 0.3 sau CNN block cuối và LSTM; 0.15 trước output |
| Batch Normalization | Sau mỗi Conv2D layer |
| SpecAugment | Mask ngẫu nhiên trên trục tần số và thời gian (p=0.5) |
| Early Stopping | patience=6, theo val_accuracy |
| Class Weighting | Tự động cân bằng lớp thiểu số |

---

## 5. Huấn luyện

### 5.1. Cấu hình huấn luyện

| Tham số | Giá trị |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 (với ReduceLROnPlateau, factor=0.5) |
| Batch size | 32 |
| Max epochs | 30 |
| Loss function | Sparse Categorical Cross-Entropy |
| Early stopping | patience=6 epochs, monitor=val_accuracy |
| Random seed | 42 (tái lập kết quả) |

### 5.2. Data Augmentation — SpecAugment

Áp dụng SpecAugment trên tập train với xác suất 50% cho mỗi loại mask:

- **Frequency masking**: Mask ngẫu nhiên tối đa 8 bin tần số liên tiếp
- **Time masking**: Mask ngẫu nhiên tối đa 16 frame thời gian liên tiếp

Mục đích: Tăng tính tổng quát hóa, giảm overfitting, mô phỏng nhiễu thực tế.

### 5.3. Quá trình huấn luyện

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

Training hội tụ ổn định, accuracy tăng đều qua các epoch.

---

## 6. Kết quả đánh giá

### 6.1. Kết quả trên tập Test

**Weighted F1-Score: 0.936 (93.6%)**

#### Classification Report (per-record):

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| abnormal_other | 0.87 | 0.84 | 0.86 | ~100 |
| normal | 0.96 | 0.97 | 0.96 | ~340 |
| **weighted avg** | **0.94** | **0.94** | **0.936** | **~440** |

### 6.2. Confusion Matrix

#### Ma trận nhầm lẫn (Records)

Mô hình phân loại đúng phần lớn các trường hợp:
- **True Normal → Predicted Normal**: Tỉ lệ cao (~97%)
- **True Abnormal → Predicted Abnormal**: Tỉ lệ tốt (~84%)
- **False Positive** (Normal dự đoán nhầm Abnormal): Thấp (~3%)
- **False Negative** (Abnormal dự đoán nhầm Normal): Cần cải thiện (~16%)

### 6.3. ROC Curve

Đường cong ROC thể hiện khả năng phân biệt giữa hai lớp. AUC (Area Under Curve) càng gần 1 càng tốt.

### 6.4. So sánh với nghiên cứu liên quan

| Nghiên cứu | Kiến trúc | Dataset | F1-Score |
|---|---|---|---|
| Deng et al. (2020) | CRNN + MFCC | PhysioNet 2016 | 90.06% |
| **IConNet (MobiSys 2024)** | CNN nhẹ | PhysioNet 2016 | 92.05% |
| Li et al. (2022) | ResNet + MFCC | PhysioNet 2016 | ~94% |
| **Mô hình của chúng tôi** | **CNN-LSTM + Log-Mel + Delta** | **PhysioNet 2016** | **93.6%** |

→ Kết quả vượt trội hơn IConNet (2024) và xấp xỉ Li et al. (2022).

---

## 7. Triển khai ứng dụng

### 7.1. Kiến trúc hệ thống

```
┌──────────────┐     HTTPS/Tunnel     ┌──────────────────┐
│  Ứng dụng    │  ◄──────────────────► │   FastAPI Server │
│  React Native│     JSON + Base64     │   + TensorFlow   │
│  (Expo)      │                       │   + Librosa      │
└──────────────┘                       └──────────────────┘
     │                                        │
     │ Upload WAV/MP3                         │ Load model
     │ Nhận kết quả                           │ Predict
     │ Hiển thị BPM                           │ Gen spectrogram
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
    "recommendation": "Nhịp tim bình thường...",
    "spectrogram_b64": "data:image/png;base64,..."
}
```

### 7.3. Giao diện ứng dụng di động

Ứng dụng hiển thị:
- **Trạng thái chẩn đoán**: Bình thường (xanh) / Bất thường (đỏ)
- **Độ chính xác**: Confidence score của model
- **BPM**: Nhịp tim ước tính bằng autocorrelation
- **Tín hiệu**: Chất lượng tín hiệu thu được
- **Phổ tần số**: Hình ảnh Log-Mel Spectrogram
- **Khuyến nghị**: Lời khuyên y tế phù hợp

---

## 8. Thảo luận

### 8.1. Điểm mạnh

1. **Kiến trúc Hybrid CNN-LSTM** kết hợp ưu điểm cả hai:
   - CNN trích xuất pattern tần số trên spectrogram
   - LSTM học nhịp điệu thời gian của nhịp tim
2. **3-channel input** (Static + Delta + Delta²) cung cấp thông tin phong phú hơn single-channel
3. **SpecAugment** hiệu quả trong việc giảm overfitting
4. **Class weighting** xử lý tốt mất cân bằng dữ liệu

### 8.2. Hạn chế

1. **Domain mismatch**: Model huấn luyện trên dữ liệu ống nghe y tế, chưa tối ưu cho mic điện thoại thông thường
2. **False Negative**: ~16% ca bất thường bị phân loại nhầm thành bình thường — cần cải thiện cho ứng dụng y tế
3. **Dữ liệu hạn chế**: Chỉ sử dụng 1 dataset (PhysioNet 2016)

### 8.3. Hướng phát triển

1. **Thu thập dữ liệu phone-specific**: Fine-tune model trên dữ liệu thu bằng mic điện thoại
2. **Stethoscope adapter**: Sử dụng phụ kiện giá rẻ để cải thiện chất lượng thu âm
3. **On-device inference**: Chuyển model sang TensorFlow Lite để chạy trực tiếp trên điện thoại
4. **Multi-class classification**: Mở rộng phân loại nhiều loại bệnh tim cụ thể

---

## 9. Kết luận

Nghiên cứu đã xây dựng thành công hệ thống phân loại âm thanh tim sử dụng mô hình **CNN-LSTM Hybrid** trên **Log-Mel Spectrogram** kết hợp **đặc trưng sai phân Delta/Delta-Delta**. Hệ thống đạt:

- **Training accuracy**: 95.0%
- **Weighted F1-score**: 93.6% (trên tập test)
- **Vượt trội** hơn IConNet (MobiSys 2024, F1=92.05%)

Hệ thống đã được triển khai thành công trên ứng dụng di động, cho phép người dùng tải file thu âm và nhận kết quả chẩn đoán với đầy đủ thông tin: BPM, chất lượng tín hiệu, phổ spectrogram, và khuyến nghị y tế.

---

## 10. Tài liệu tham khảo

1. Clifford, G.D., et al. (2016). "Classification of normal/abnormal heart sound recordings: the PhysioNet/Computing in Cardiology Challenge 2016." *Computing in Cardiology*, 43, 609-612.

2. Vu, L., & Tran, T. (2024). "Detecting abnormal heart sound using mobile phones and on-device IConNet." *arXiv:2412.03267*. MobiSys 2024.

3. Makimoto, H., et al. (2022). "Efficient screening for severe aortic valve stenosis using understandable artificial intelligence." *European Heart Journal – Digital Health*, 3(2), 165-173.

4. Deng, M., Meng, T., et al. (2020). "Heart sound classification based on improved MFCC features and convolutional recurrent neural networks." *Neural Networks*, 130, 22-32.

5. Li, F., Zhang, Z., Wang, L., & Liu, W. (2022). "Heart sound classification based on improved mel-frequency spectral coefficients and deep residual learning." *Frontiers in Physiology*, 13, 1084420.

6. Park, D.S., et al. (2019). "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition." *Proc. Interspeech 2019*.

---

## Phụ lục

### A. Cấu trúc dự án

```
heart-sound-classification/
├── data/
│   ├── raw/physionet_2016/     # Dataset gốc
│   ├── processed/logmels/      # Features đã trích xuất
│   └── metadata/               # Label maps, splits, norm stats
├── models/tf_heart_sound/      # Model đã train
├── src/
│   ├── preprocess/
│   │   └── extract_features.py # Trích xuất Log-Mel + Delta
│   ├── training/
│   │   ├── model_cnn_lstm.py   # Kiến trúc CNN-LSTM
│   │   └── train_tf.py         # Script huấn luyện
│   ├── inference/
│   │   ├── audio_utils.py      # Preprocessing cho inference
│   │   └── predict_tf.py       # Dự đoán + BPM + Signal Quality
│   └── app/
│       └── main.py             # FastAPI server
├── reports/figures/            # Confusion matrices, ROC curve
└── heart-sound-app/            # React Native mobile app
```

### B. Môi trường thực nghiệm

| Thành phần | Chi tiết |
|---|---|
| Ngôn ngữ | Python 3.11 |
| Framework ML | TensorFlow 2.16 / Keras 3 |
| Xử lý âm thanh | Librosa, SciPy |
| API Server | FastAPI + Uvicorn |
| Mobile App | React Native (Expo) |
| Hệ điều hành | Windows 11 |
