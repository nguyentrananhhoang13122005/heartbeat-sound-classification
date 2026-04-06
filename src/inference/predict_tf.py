import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from .audio_utils import (
    load_audio, segment_audio, logmel,
    SR, SEGMENT_SECONDS, HOP_SECONDS
)

def estimate_signal_quality(y: np.ndarray, sr: int = SR) -> float:
    """Estimate heart sound signal quality.
    Returns a score 0-1.  Low score => likely noise / poor contact.
    Uses the ratio of energy in heart-sound band (25-150 Hz) vs total energy.
    """
    from scipy.signal import butter, sosfilt
    # Heart sound fundamental band: 25-150 Hz
    nyq = sr / 2.0
    sos = butter(4, [25 / nyq, min(150 / nyq, 0.99)], btype='band', output='sos')
    y_heart = sosfilt(sos, y)
    energy_heart = np.sum(y_heart ** 2)
    energy_total = np.sum(y ** 2) + 1e-10
    ratio = energy_heart / energy_total
    # Typical stethoscope recording: ratio > 0.3
    # Phone mic with poor contact: ratio < 0.05
    quality = float(np.clip(ratio / 0.3, 0.0, 1.0))
    return quality


def estimate_bpm(y: np.ndarray, sr: int = SR) -> int:
    """Estimate heart rate (BPM) using autocorrelation on bandpass-filtered signal."""
    from scipy.signal import butter, sosfilt
    nyq = sr / 2.0
    # Filter to heart S1/S2 frequency band (20-200 Hz)
    sos = butter(4, [20 / nyq, min(200 / nyq, 0.99)], btype='band', output='sos')
    y_filt = sosfilt(sos, y)

    # Envelope via squaring + smoothing
    envelope = y_filt ** 2
    win = int(0.05 * sr)  # 50ms smoothing window
    if win > 0:
        kernel = np.ones(win) / win
        envelope = np.convolve(envelope, kernel, mode='same')

    # Autocorrelation
    envelope = envelope - np.mean(envelope)
    corr = np.correlate(envelope, envelope, mode='full')
    corr = corr[len(corr) // 2:]  # positive lags only
    corr = corr / (corr[0] + 1e-10)  # normalize

    # Find first peak after minimum lag (40 BPM = 1.5s, 200 BPM = 0.3s)
    min_lag = int(0.3 * sr)   # 200 BPM max
    max_lag = int(1.5 * sr)   # 40 BPM min
    if max_lag > len(corr):
        max_lag = len(corr)
    if min_lag >= max_lag:
        return 72  # default fallback

    search = corr[min_lag:max_lag]
    if len(search) == 0:
        return 72

    peak_lag = min_lag + int(np.argmax(search))
    if peak_lag == 0:
        return 72

    bpm = int(60.0 * sr / peak_lag)
    # Clamp to reasonable range
    bpm = max(40, min(200, bpm))
    return bpm


def get_recommendation(prediction: str, confidence: float, quality: float, bpm: int) -> dict:
    """Generate medical recommendation based on results."""
    if quality < 0.3:
        return {
            "level": "warning",
            "title": "Chất lượng tín hiệu kém",
            "message": "Không thu được tín hiệu tim rõ ràng. Vui lòng đo lại trong phòng yên tĩnh, đặt điện thoại áp sát ngực trái.",
            "icon": "🔄"
        }
    elif quality < 0.5:
        return {
            "level": "info",
            "title": "Tín hiệu yếu",
            "message": "Tín hiệu tim thu được chưa rõ. Kết quả mang tính tham khảo. Nên đo lại hoặc sử dụng ống nghe chuyên dụng.",
            "icon": "⚠️"
        }
    elif prediction == "normal" and confidence > 0.7:
        return {
            "level": "success",
            "title": "Nhịp tim bình thường",
            "message": f"Nhịp tim {bpm} BPM, nằm trong phạm vi bình thường. Tiếp tục duy trì lối sống lành mạnh.",
            "icon": "✅"
        }
    elif prediction == "abnormal_other":
        return {
            "level": "danger",
            "title": "Phát hiện dấu hiệu bất thường",
            "message": f"Nhịp tim {bpm} BPM. Hệ thống phát hiện dấu hiệu bất thường trong âm thanh tim. Khuyến nghị đến cơ sở y tế để kiểm tra chuyên sâu.",
            "icon": "🏥"
        }
    else:
        return {
            "level": "info",
            "title": "Kết quả chưa rõ ràng",
            "message": f"Nhịp tim {bpm} BPM. Độ tin cậy của kết quả chưa cao. Nên đo lại hoặc tham khảo ý kiến bác sĩ.",
            "icon": "ℹ️"
        }

class Predictor:
    def __init__(self, model_path: Path, label_map_path: Path, norm_path: Path, abnormal_threshold: float = 0.7):
        self.model = tf.keras.models.load_model(str(model_path), compile=False)
        with open(label_map_path, "r") as f:
            lm = json.load(f)
        labels = lm["labels"]
        l2i = {k: int(v) for k, v in lm["label_to_idx"].items()}
        # Make sure idx_to_label aligns with model outputs
        self.idx_to_label = [None] * len(labels)
        for k, v in l2i.items():
            self.idx_to_label[v] = k

        with open(norm_path, "r") as f:
            stats = json.load(f)
        self.mean = float(stats["global_mean"])
        self.std = float(stats["global_std"])
        self.abnormal_threshold = float(abnormal_threshold)

    def _prep_batch(self, feats: List[np.ndarray]) -> np.ndarray:
        # feats: list of [n_mels, T, 3] (static + delta + delta-delta)
        # Normalize and stack; pad to same T if needed
        T_max = max(x.shape[1] for x in feats)
        batch = []
        for x in feats:
            if x.shape[1] < T_max:
                pad = np.zeros((x.shape[0], T_max - x.shape[1], x.shape[2]), dtype=np.float32)
                x = np.concatenate([x, pad], axis=1)
            x = (x - self.mean) / (self.std + 1e-8)
            batch.append(x)
        return np.stack(batch, axis=0)  # [B, n_mels, T, 3]

    def predict_file(self, audio_path: Path) -> Dict:
        import librosa as _librosa

        # --- 1) Load RAW audio (before normalization) for signal quality ---
        y_raw, _ = _librosa.load(str(audio_path), sr=SR, mono=True)
        if y_raw.size == 0:
            y_raw = np.zeros(int(SEGMENT_SECONDS * SR), dtype=np.float32)
        y_raw = y_raw - np.mean(y_raw)  # DC offset only
        quality = estimate_signal_quality(y_raw, sr=SR)

        # --- 2) Load processed audio (normalized + filtered) for prediction ---
        y = load_audio(audio_path)
        segs = segment_audio(y, sr=SR, seg_sec=SEGMENT_SECONDS, hop_sec=HOP_SECONDS)
        if not segs:
            raise ValueError("No segments produced from audio; check the file.")

        # Compute features for each segment
        feats = []
        for (s, e, s_sec, e_sec) in segs:
            y_seg = y[s:e]
            feats.append(logmel(y_seg, sr=SR))

        batch = self._prep_batch(feats)
        probs = self.model.predict(batch, verbose=0)  # [B, C]
        probs = probs.astype(np.float32)

        # --- 3) OOD Detection via segment consistency + low signal quality ---
        # Only flag OOD when signal quality is low (phone mic, not stethoscope)
        # AND all segments uniformly predict abnormal.
        # Stethoscope recordings have quality > 0.7, phone mic < 0.5
        ab_idx = self.idx_to_label.index("abnormal_other") if "abnormal_other" in self.idx_to_label else None
        ood_detected = False
        if ab_idx is not None and len(probs) > 1 and quality < 0.7:
            ab_probs_per_seg = probs[:, ab_idx]
            seg_std = float(np.std(ab_probs_per_seg))
            seg_mean = float(np.mean(ab_probs_per_seg))
            # OOD: low quality + all segments say abnormal >85% with std < 0.08
            if seg_mean > 0.85 and seg_std < 0.08:
                ood_detected = True

        # --- 4) Aggregate probs and apply calibration ---
        rec_probs = probs.mean(axis=0)

        if ab_idx is not None:
            # Calibration factor based on signal quality + OOD detection
            if ood_detected:
                # Strong blend towards 50/50 — likely phone noise
                blend = 0.3  # only keep 30% of model prediction
                uniform = np.ones_like(rec_probs) / len(rec_probs)
                rec_probs = blend * rec_probs + (1 - blend) * uniform
                quality = min(quality, 0.3)  # cap quality for display
            elif quality < 0.5:
                # Low signal quality — blend towards 50/50
                blend = quality / 0.5
                uniform = np.ones_like(rec_probs) / len(rec_probs)
                rec_probs = blend * rec_probs + (1 - blend) * uniform

        pred_idx = int(np.argmax(rec_probs))
        pred_label = self.idx_to_label[pred_idx]
        confidence = float(rec_probs[pred_idx])

        # Segment-level details
        seg_details = []
        for i, ((s, e, s_sec, e_sec), p) in enumerate(zip(segs, probs)):
            top_i = int(np.argmax(p))
            top_label = self.idx_to_label[top_i]
            seg_details.append({
                "segment_idx": i,
                "start_sec": float(s_sec),
                "end_sec": float(e_sec),
                "top_label": top_label,
                "probs": {self.idx_to_label[j]: float(p[j]) for j in range(len(self.idx_to_label))}
            })

        # Highlight segments likely abnormal (if abnormal class exists)
        highlight_idxs = []
        if ab_idx is not None:
            for i, p in enumerate(probs):
                if p[ab_idx] >= self.abnormal_threshold:
                    highlight_idxs.append(i)

        # Estimate BPM and generate recommendation
        bpm = estimate_bpm(y_raw, sr=SR)  # use raw audio for BPM
        recommendation = get_recommendation(pred_label, confidence, quality, bpm)

        # Generate spectrogram image as base64
        from src.inference.audio_utils import render_spectrogram_base64
        spectrogram_b64 = render_spectrogram_base64(y, sr=SR)

        return {
            "record": {
                "primary_prediction": pred_label,
                "confidence": confidence,
                "probs": {self.idx_to_label[j]: float(rec_probs[j]) for j in range(len(self.idx_to_label))}
            },
            "bpm": bpm,
            "signal_quality": quality,
            "recommendation": recommendation,
            "spectrogram_b64": spectrogram_b64,
            "segments": seg_details,
            "segment_seconds": {"length": SEGMENT_SECONDS, "hop": HOP_SECONDS},
            "highlight_segments": highlight_idxs,
        }