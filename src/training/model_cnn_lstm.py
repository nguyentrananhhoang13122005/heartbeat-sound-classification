"""
CNN-LSTM hybrid model for heart sound classification.

Architecture:
  Input (Log-Mel Spectrogram) [n_mels, T, 1]
  -> CNN blocks extract local frequency features per time-frame
  -> Reshape to sequence [time_steps, features]
  -> Bidirectional LSTM captures temporal heartbeat rhythm (S1 -> S2)
  -> Dense softmax for classification
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_lstm_model(
    n_mels: int,
    num_classes: int,
    dropout: float = 0.3,
    lr: float = 1e-3,
    lstm_units: int = 64,
) -> tf.keras.Model:
    """
    Build a CNN-LSTM hybrid model.

    CNN extracts spatial/frequency features from the spectrogram,
    then LSTM learns temporal dependencies across time frames
    (e.g., S1 -> pause -> S2 -> pause heartbeat rhythm).

    Args:
        n_mels: Number of mel frequency bins (height of spectrogram).
        num_classes: Number of output classes.
        dropout: Dropout rate for regularization.
        lr: Learning rate for Adam optimizer.
        lstm_units: Number of LSTM hidden units.
    """
    inputs = tf.keras.Input(shape=(n_mels, None, 3), name="logmel_input")

    # ---- CNN Feature Extractor ----
    # Block 1: Extract low-level frequency patterns
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Block 2: Extract mid-level frequency patterns
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Block 3: Extract higher-level frequency patterns
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 1))(x)
    # Pool only along frequency axis (2,1) to preserve temporal resolution

    x = layers.Dropout(dropout)(x)

    # ---- Reshape: [batch, freq, time, channels] -> [batch, time, freq*channels] ----
    # After CNN: shape is [batch, n_mels/8, T', 128]
    # We want to treat T' as the sequence length for LSTM
    # Permute to [batch, T', n_mels/8, 128] then flatten freq*channels
    x = layers.Permute((2, 1, 3))(x)  # [batch, T', freq_reduced, 128]

    # Use Reshape with -1 to dynamically flatten the last two dims
    x = layers.TimeDistributed(layers.Flatten())(x)  # [batch, T', freq_reduced * 128]

    # ---- LSTM Temporal Sequence Learner ----
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=False)
    )(x)
    x = layers.Dropout(dropout)(x)

    # ---- Classification Head ----
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout / 2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="heart_sound_cnn_lstm")

    optimizer = tf.keras.optimizers.Adam(learning_rate=float(lr))
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model
