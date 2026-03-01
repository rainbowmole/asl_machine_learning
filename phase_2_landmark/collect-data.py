import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# --- MediaPipe setup ---
model_path = 'hand_landmarker.task'
base_options = mp_python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)
detector = vision.HandLandmarker.create_from_options(options)

# --- Labels ---
labels = ['HAPPY', 'GOOD', 'LOVE', 'EXCITED', 'SAD', 'ANGRY', 'SCARED', 'HATE', 'OKAY', 'FEEL']

# --- Settings ---
SEQUENCE_LENGTH = 30
SAMPLES_PER_LABEL = 100
DATA_DIR = 'landmark_data'
# Fixed frame size: 63 left + 63 right + 1 left_present + 1 right_present = 128
FRAME_SIZE = 128

os.makedirs(DATA_DIR, exist_ok=True)
for label in labels:
    os.makedirs(os.path.join(DATA_DIR, label), exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

current_label_idx = 0
state = 'WAITING'
sequence = []
samples_collected = 0
countdown_start = 0

print(f"Starting collection | Word: {labels[current_label_idx]}")
print("SPACE=Record  N=Next  B=Back  Q=Quit")

def extract_landmarks(hand_landmarks):
    """Normalize landmarks relative to wrist."""
    wrist = hand_landmarks[0]
    coords = []
    for lm in hand_landmarks:
        coords.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z
        ])
    return coords  # 63 values

def build_frame(left_coords, right_coords):
    """
    Always builds a fixed 128-value frame:
    [left 63] + [right 63] + [left_present] + [right_present]
    """
    left_data = left_coords if left_coords else [0.0] * 63
    right_data = right_coords if right_coords else [0.0] * 63
    left_present = 1.0 if left_coords else 0.0
    right_present = 1.0 if right_coords else 0.0
    return left_data + right_data + [left_present, right_present]

def negate_x(coords):
    """Negate the X component of every landmark (x,y,z triplets)."""
    out = list(coords)
    for i in range(0, len(out), 3):
        out[i] = -out[i]
    return out

def mirror_sequence(sequence):
    """
    Produces a horizontally-mirrored copy of a sequence.
    Left and right hand slots are swapped and their X coordinates negated,
    so a gesture recorded with one hand also trains the other.
    """
    mirrored = []
    for frame in sequence:
        frame = list(frame)
        left_coords   = frame[0:63]
        right_coords  = frame[63:126]
        left_present  = frame[126]
        right_present = frame[127]

        new_left  = negate_x(right_coords) if right_present else [0.0] * 63
        new_right = negate_x(left_coords)  if left_present  else [0.0] * 63
        mirrored.append(new_left + new_right + [right_present, left_present])
    return mirrored

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )
    result = detector.detect(mp_image)

    left_coords = None
    right_coords = None
    left_detected = False
    right_detected = False

    if result.hand_landmarks and result.handedness:
        for i, handedness in enumerate(result.handedness):
            raw_label = handedness[0].display_name
            # Correct for mirror flip
            corrected = 'Left' if raw_label == 'Right' else 'Right'
            hand_landmarks = result.hand_landmarks[i]

            if corrected == 'Left':
                left_detected = True
                left_coords = extract_landmarks(hand_landmarks)
                # Draw blue dots for left hand
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (255, 100, 100), -1)

            elif corrected == 'Right':
                right_detected = True
                right_coords = extract_landmarks(hand_landmarks)
                # Draw red dots for right hand
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (100, 100, 255), -1)

    hand_detected = left_detected or right_detected
    frame_data = build_frame(left_coords, right_coords)

    # --- State machine ---
    if state == 'COUNTDOWN':
        elapsed = (cv2.getTickCount() - countdown_start) / cv2.getTickFrequency()
        if elapsed >= 3:
            state = 'COLLECTING'
            sequence = []

    elif state == 'COLLECTING':
        if hand_detected:
            sequence.append(frame_data)
            if len(sequence) == SEQUENCE_LENGTH:
                sample_path = os.path.join(
                    DATA_DIR, labels[current_label_idx],
                    f'seq_{samples_collected:04d}.npy'
                )
                np.save(sample_path, np.array(sequence))

                # --- Mirror augmentation: swap hands so either hand works ---
                mirror_path = os.path.join(
                    DATA_DIR, labels[current_label_idx],
                    f'seq_{samples_collected:04d}_mirror.npy'
                )
                np.save(mirror_path, np.array(mirror_sequence(sequence)))

                samples_collected += 1
                sequence = []
                if samples_collected >= SAMPLES_PER_LABEL:
                    state = 'DONE'
                    print(f"✓ {labels[current_label_idx]} done! Press N.")
                else:
                    state = 'WAITING'
        else:
            sequence = []
            state = 'WAITING'

    # --- Minimal UI ---
    bar = frame.copy()
    cv2.rectangle(bar, (0, h - 80), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.6, frame, 0.4, 0, frame)

    # Word + count top left
    cv2.putText(frame, labels[current_label_idx],
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    cv2.putText(frame, f'{samples_collected}/{SAMPLES_PER_LABEL}',
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Hand indicators top left
    left_color = (255, 100, 100) if left_detected else (60, 60, 60)
    right_color = (100, 100, 255) if right_detected else (60, 60, 60)
    cv2.putText(frame, 'L', (10, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, left_color, 2)
    cv2.putText(frame, 'R', (30, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, right_color, 2)

    # Status top right
    if state == 'WAITING':
        status = 'READY'
        s_color = (0, 255, 0)
    elif state == 'COUNTDOWN':
        elapsed = (cv2.getTickCount() - countdown_start) / cv2.getTickFrequency()
        status = str(3 - int(elapsed))
        s_color = (0, 165, 255)
    elif state == 'COLLECTING':
        status = f'REC {len(sequence)}/{SEQUENCE_LENGTH}'
        s_color = (0, 0, 255)
    elif state == 'DONE':
        status = 'DONE'
        s_color = (0, 255, 0)

    text_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.putText(frame, status,
                (w - text_size[0] - 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, s_color, 2)

    # Progress bar
    progress = int((samples_collected / SAMPLES_PER_LABEL) * (w - 40))
    cv2.rectangle(frame, (20, h - 65), (w - 20, h - 55), (50, 50, 50), -1)
    cv2.rectangle(frame, (20, h - 65), (20 + progress, h - 55), (0, 255, 0), -1)

    # Word list bottom
    for i, label in enumerate(labels):
        color = (0, 255, 0) if i < current_label_idx else \
                (255, 255, 0) if i == current_label_idx else (80, 80, 80)
        cv2.putText(frame, label,
                    (10 + i * 70, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # Recording flash dot
    if state == 'COLLECTING':
        cv2.circle(frame, (w - 20, 55), 8, (0, 0, 255), -1)

    cv2.imshow('Phase 2 - Data Collection', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        if state == 'WAITING' and hand_detected:
            state = 'COUNTDOWN'
            countdown_start = cv2.getTickCount()
        elif not hand_detected:
            print("No hand detected!")
    elif key == ord('n'):
        if current_label_idx < len(labels) - 1:
            current_label_idx += 1
            samples_collected = 0
            state = 'WAITING'
            sequence = []
            print(f"Next: {labels[current_label_idx]}")
    elif key == ord('b'):
        if current_label_idx > 0:
            current_label_idx -= 1
            samples_collected = 0
            state = 'WAITING'
            sequence = []
            print(f"Back: {labels[current_label_idx]}")

cap.release()
cv2.destroyAllWindows()
print("Data collection complete!")