import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- Mirror augmentation ---
def mirror_landmarks(frame_data):
    """
    Mirror right hand to generate synthetic left hand data.
    Frame format: [left 63] + [right 63] + [left_present] + [right_present] = 128
    """
    left_coords = frame_data[:63]
    right_coords = frame_data[63:126]
    left_present = frame_data[126]
    right_present = frame_data[127]

    # Mirror left hand (negate x)
    mirrored_right = []
    for i in range(21):
        x = left_coords[i*3]
        y = left_coords[i*3 + 1]
        z = left_coords[i*3 + 2]
        mirrored_right.extend([-x, y, z])

    # Mirror right hand (negate x)
    mirrored_left = []
    for i in range(21):
        x = right_coords[i*3]
        y = right_coords[i*3 + 1]
        z = right_coords[i*3 + 2]
        mirrored_left.extend([-x, y, z])

    # Swap hands — left becomes right and vice versa
    return mirrored_left + mirrored_right + [right_present, left_present]

def mirror_sequence(sequence):
    """Mirror an entire sequence of frames."""
    return [mirror_landmarks(frame) for frame in sequence]

# --- Load sequences ---
labels = ['HAPPY', 'GOOD', 'LOVE', 'EXCITED', 'SAD', 'ANGRY', 'SCARED', 'HATE', 'OKAY', 'FEEL']
DATA_DIR = 'landmark_data'

X = []
y = []

print("Loading data...")
for label in labels:
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.exists(label_dir):
        print(f"Warning: No data found for {label}")
        continue
    files = [f for f in os.listdir(label_dir) if f.endswith('.npy')]
    print(f"{label}: {len(files)} sequences")
    for file in files:
        sequence = np.load(os.path.join(label_dir, file))
        X.append(sequence)
        y.append(label)

X_raw = np.array(X)
y_raw = np.array(y)

print(f"\nOriginal sequences: {len(X_raw)}")

# --- Apply mirror augmentation ---
print("Applying mirror augmentation...")
X_augmented = []
y_augmented = []

for sequence, label in zip(X_raw, y_raw):
    # Keep original
    X_augmented.append(sequence)
    y_augmented.append(label)

    # Add mirrored version
    mirrored = mirror_sequence(sequence)
    X_augmented.append(mirrored)
    y_augmented.append(label)

X = np.array(X_augmented)
y = np.array(y_augmented)

print(f"After augmentation: {len(X)} sequences (2x original)")
print(f"Sequence shape: {X.shape}")

# --- Encode labels ---
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = tf.keras.utils.to_categorical(y_encoded)
np.save('landmark_data/classes.npy', encoder.classes_)

# --- Split ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# --- Build LSTM model ---
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(30, 128)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(encoder.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- Train ---
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            patience=15,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=5,
            factor=0.5
        )
    ]
)

model.save('landmark_model.h5')
print("\nModel saved as landmark_model.h5")