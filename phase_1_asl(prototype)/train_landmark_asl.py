"""
Train a landmark-based ASL letter classifier.
Input:  asl_landmarks.csv  (produced by extract_landmarks.py)
        95 features per sample
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
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)

CSV_PATH   = 'asl_landmarks.csv'
MODEL_OUT  = 'asl_landmark_model.keras'
LABELS_OUT = 'label_classes.npy'

# --- Load ---
print("Loading CSV...")
df = pd.read_csv(CSV_PATH)
df = df[df['label'] != 'nothing']   # remove nothing class
print(f"Total samples: {len(df)}")
print(df['label'].value_counts().to_string())

X     = df.drop(columns=['label']).values.astype(np.float32)
y_raw = df['label'].values

NUM_FEATURES = X.shape[1]
print(f"\nFeatures per sample: {NUM_FEATURES}")

le = LabelEncoder()
y  = le.fit_transform(y_raw)
np.save(LABELS_OUT, le.classes_)
print(f"Classes ({len(le.classes_)}): {le.classes_}")

NUM_CLASSES = len(le.classes_)

# --- Split 70/15/15 ---
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# --- Class weights ---
class_weights     = compute_class_weight(
    class_weight='balanced', classes=np.unique(y_train), y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

# --- Augmentation ---
def augment_noise(X_batch, scale=0.005):
    return X_batch + np.random.normal(0, scale, X_batch.shape).astype(np.float32)

def augment_scale(X_batch, scale_range=(0.90, 1.10)):
    X_out  = X_batch.copy()
    scales = np.random.uniform(
        scale_range[0], scale_range[1], size=(len(X_batch), 1)
    ).astype(np.float32)
    X_out[:, :63] *= scales
    return X_out

def mirror_X(X_batch):
    """
    Mirror along X — makes model hand-agnostic.
    Feature layout (95):
      [0:63]  landmark coords — negate every x (indices 0,3,6,...,60)
      [63:66] palm normal     — negate nx (index 63)
      [66:]   scalars         — unchanged
    """
    X_m = X_batch.copy()
    X_m[:, 0:63:3] *= -1
    if X_m.shape[1] > 63:
        X_m[:, 63] *= -1
    return X_m

print("\nApplying augmentation (6x)...")
X_mirror = mirror_X(X_train)
X_train_final = np.concatenate([
    X_train,
    augment_noise(X_train),
    augment_scale(X_train),
    X_mirror,
    augment_noise(X_mirror),
    augment_scale(X_mirror),
])
y_train_final = np.concatenate([y_train] * 6)
print(f"After augmentation: {len(X_train_final)} samples")

# --- Model ---
model = tf.keras.Sequential([
    layers.Input(shape=(NUM_FEATURES,)),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.35),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.30),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.25),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.20),
    layers.Dense(NUM_CLASSES, activation='softmax')
], name='asl_landmark_classifier')

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(patience=8, restore_best_weights=True, monitor='val_accuracy'),
    ReduceLROnPlateau(factor=0.5, patience=4, monitor='val_loss',
                      verbose=1, min_lr=1e-6),
    ModelCheckpoint(MODEL_OUT, save_best_only=True,
                    monitor='val_accuracy', verbose=1)
]

# --- Train ---
print("\nTraining...")
model.fit(
    X_train_final, y_train_final,
    validation_data=(X_val, y_val),
    epochs=80,
    batch_size=128,
    callbacks=callbacks,
    class_weight=class_weight_dict
)

# --- Evaluate ---
val_loss,  val_acc  = model.evaluate(X_val,  y_val,  verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n{'='*40}")
print(f"Val  accuracy: {val_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

# Per-class report
print(f"\nPer-class accuracy on test set:")
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
for i, cls in enumerate(le.classes_):
    mask    = y_test == i
    if mask.sum() == 0:
        continue
    cls_acc = (y_pred[mask] == i).mean()
    status  = "✓" if cls_acc >= 0.90 else "✗"
    print(f"  {status} {cls:8s}  {cls_acc:.2%}  ({mask.sum()} samples)")

print(f"\nModel saved to {MODEL_OUT}")
print(f"Labels saved to {LABELS_OUT}")