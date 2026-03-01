"""
Train a landmark-based ASL letter classifier.
Input: asl_landmarks.csv  (produced by extract_landmarks.py)
       71 features per sample:
         63 scale-normalised landmark coords
          3 palm normal vector
          5 finger curl distances
Output: asl_landmark_model.keras + label_classes.npy

Run from: phase_1_asl(prototype)/
  python train_landmark_asl.py
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

CSV_PATH   = 'asl_landmarks.csv'
MODEL_OUT  = 'asl_landmark_model.keras'
LABELS_OUT = 'label_classes.npy'

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading CSV...")
df = pd.read_csv(CSV_PATH)
print(f"Total samples: {len(df)}")
print(df['label'].value_counts().to_string())

X = df.drop(columns=['label']).values.astype(np.float32)  # (N, 63)
y_raw = df['label'].values

le = LabelEncoder()
y = le.fit_transform(y_raw)                               # integer labels
np.save(LABELS_OUT, le.classes_)
print(f"\nClasses ({len(le.classes_)}): {le.classes_}")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} | Val: {len(X_val)}")

NUM_CLASSES = len(le.classes_)

# ── Augmentation ──────────────────────────────────────────────────────────────
def augment_noise(X_batch, y_batch):
    noise = np.random.normal(0, 0.005, X_batch.shape).astype(np.float32)
    return X_batch + noise, y_batch

def mirror_X(X_batch):
    """
    Mirror each sample along the X axis so the model recognises gestures
    performed with either hand.

    Feature layout (per row):
      [0:63]  – 63 landmark coords as (x, y, z) triplets for 21 landmarks
      [63:66] – palm normal vector (nx, ny, nz)
      [66:71] – finger curl distances (scalars, unchanged under mirroring)

    We negate:
      • every X component in the 63-landmark block  (indices 0, 3, 6, …, 60)
      • the palm-normal X component                  (index 63)
    """
    X_mirror = X_batch.copy()
    # Negate X of the 21 landmark triplets (indices 0,3,6,...,60)
    X_mirror[:, 0:63:3] *= -1
    # Negate palm-normal X if the feature exists
    if X_mirror.shape[1] > 63:
        X_mirror[:, 63] *= -1
    return X_mirror

X_aug, y_aug     = augment_noise(X_train, y_train)
X_mirror         = mirror_X(X_train)
X_mirror_aug, _  = augment_noise(X_mirror, y_train)  # also noise-augment the mirror

X_train_final = np.concatenate([X_train, X_aug, X_mirror, X_mirror_aug])
y_train_final = np.concatenate([y_train, y_aug, y_train,  y_train])
print(f"After augmentation: {len(X_train_final)} training samples "
      f"(original + noise + mirror + mirror+noise)")

# ── Model ──────────────────────────────────────────────────────────────────────
NUM_FEATURES = X.shape[1]  # 71 (auto-detected from CSV)
model = tf.keras.Sequential([
    layers.Input(shape=(NUM_FEATURES,)),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(NUM_CLASSES, activation='softmax')
], name='asl_landmark_classifier')

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(patience=6, restore_best_weights=True, monitor='val_accuracy'),
    ReduceLROnPlateau(factor=0.5, patience=3, monitor='val_loss', verbose=1),
    ModelCheckpoint(MODEL_OUT, save_best_only=True, monitor='val_accuracy', verbose=1)
]

model.fit(
    X_train_final, y_train_final,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=256,
    callbacks=callbacks
)

# ── Evaluate ───────────────────────────────────────────────────────────────────
loss, acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\nFinal val accuracy: {acc:.4f}")
print(f"Model saved to {MODEL_OUT}")
print(f"Labels saved to {LABELS_OUT}")
