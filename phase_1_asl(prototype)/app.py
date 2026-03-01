import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
from collections import deque
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

MODEL_PATH  = 'asl_landmark_model.keras'
LABELS_PATH = 'label_classes.npy'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Run train_landmark_asl.py first.")
if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"Labels file '{LABELS_PATH}' not found. Run train_landmark_asl.py first.")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
labels = list(np.load(LABELS_PATH, allow_pickle=True))
print(f"Model ready! {len(labels)} classes: {labels}")
print("Controls: [A] Add letter (manual) | [S] Analyze sentiment | [C] Clear | [Q] Quit")
print("Auto-mode: hold a letter steady for 1.5 s to add it automatically.")
print("           Remove hand for 1.5 s to auto-insert a space (end of word).")
analyzer = SentimentIntensityAnalyzer()

# ── Landmark index constants (must match extract_landmarks.py) ─────────────
_TIPS = [4,  8, 12, 16, 20]
_MCPS = [2,  5,  9, 13, 17]
_TIP_PAIRS   = [(4,8),(4,12),(4,16),(4,20),
                (8,12),(8,16),(8,20),
                (12,16),(12,20),(16,20)]
_PIP_TRIPLETS = [(1,2,3),(5,6,7),(9,10,11),(13,14,15),(17,18,19)]
_DIP_TRIPLETS = [(2,3,4),(6,7,8),(10,11,12),(14,15,16),(18,19,20)]

def _joint_angle(raw, p, j, n):
    v1 = raw[p] - raw[j];  v2 = raw[n] - raw[j]
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return float(np.arccos(np.clip(cos_a, -1.0, 1.0)))

def extract_features(lm):
    """
    95-feature vector — must match extract_landmarks.py exactly:
      63  scale-normalised coords
       3  palm normal
       5  finger curl (tip-to-MCP)
       5  PIP joint angles   (middle-knuckle bend)
       5  DIP joint angles   (fingertip-knuckle bend)
      10  pairwise fingertip distances
       4  thumb-tip to fingertip distances
    """
    wx, wy, wz = lm[0].x, lm[0].y, lm[0].z
    raw = np.array([(p.x-wx, p.y-wy, p.z-wz) for p in lm], dtype=np.float32)
    scale = np.linalg.norm(raw[9]) + 1e-6
    scaled     = (raw / scale).flatten()
    normal     = np.cross(raw[5], raw[17])
    normal    /= np.linalg.norm(normal) + 1e-6
    curls      = [np.linalg.norm(raw[t]-raw[m])/scale for t,m in zip(_TIPS,_MCPS)]
    pip_angles = [_joint_angle(raw,p,j,n) for p,j,n in _PIP_TRIPLETS]
    dip_angles = [_joint_angle(raw,p,j,n) for p,j,n in _DIP_TRIPLETS]
    pair_dists = [np.linalg.norm(raw[a]-raw[b])/scale for a,b in _TIP_PAIRS]
    thumb_dist = [np.linalg.norm(raw[4]-raw[t])/scale for t in [8,12,16,20]]
    return np.concatenate(
        [scaled, normal, curls, pip_angles, dip_angles, pair_dists, thumb_dist]
    ).astype(np.float32)  # 95

# --- MediaPipe hand tracking setup ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# --- Camera setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

sentence = ""
sentiment_display = ""
letter = "?"
confidence = 0.0

BUFFER_SIZE = 15
prediction_buffer = []
CONFIDENCE_THRESHOLD = 0.82

# ── Auto-add (dwell) settings ─────────────────────────────────────────────────
DWELL_TIME       = 1.5   # seconds a confident letter must be held to auto-add
COOLDOWN_TIME    = 0.9   # seconds before the same letter can be added again
NO_HAND_SPACE_T  = 1.5   # seconds without a hand before a space is auto-inserted

dwell_letter      = None   # which letter the dwell timer is currently tracking
dwell_start       = None   # time.time() when the current dwell started
last_added_letter = None   # last letter that was added (for cooldown)
last_add_time     = 0.0    # time.time() when a letter was last added
no_hand_since     = None   # time.time() when hand was last lost

# ── J / Z motion detection ────────────────────────────────────────────────────
MOTION_FRAMES = 25
motion_history = deque(maxlen=MOTION_FRAMES)
# J: pinky tip drops DOWN in image (y increases) while rest of hand holds 'I'
# Z: index tip sweeps horizontally with enough range (drawing a Z)
J_DROP_PX   = 28   # pixels pinky tip must drop to confirm J
Z_SWEEP_PX  = 55   # pixels index tip must sweep horizontally to confirm Z

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # --- MediaPipe hand detection ---
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    hand_detected = False

    if result.multi_hand_landmarks:
        hand_detected = True
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        lm = hand_landmarks.landmark

        # --- 95-feature vector (matches training) ---
        feat = extract_features(lm).reshape(1, -1)  # (1, 95)

        prediction = model(feat, training=False).numpy()[0]
        confidence = np.max(prediction)

        # Probability-averaging buffer (more stable than mode vote)
        prediction_buffer.append(prediction)
        if len(prediction_buffer) > BUFFER_SIZE:
            prediction_buffer.pop(0)
        avg_probs = np.mean(prediction_buffer, axis=0)
        letter = labels[np.argmax(avg_probs)]
        confidence = np.max(avg_probs)

        # --- J / Z motion detection (pixel coordinates) ---
        # Use raw image pixel positions — avoids normalisation ambiguity
        pinky_y_px = lm[20].y * h  # y increases downward in image
        index_x_px = lm[8].x  * w  # x in pixels
        motion_history.append((pinky_y_px, index_x_px))

        if len(motion_history) == MOTION_FRAMES:
            pinky_ys = [m[0] for m in motion_history]
            index_xs = [m[1] for m in motion_history]
            pinky_drop  = max(pinky_ys) - pinky_ys[0]   # how far pinky fell
            index_sweep = max(index_xs) - min(index_xs)  # horizontal range

            # J: static model reads 'I' + pinky hooks downward
            if letter == 'I' and pinky_drop > J_DROP_PX:
                letter = 'J'
            # Z: static model reads index-extended letter + large horizontal sweep
            elif letter in ('G', 'D', 'X', 'I') and index_sweep > Z_SWEEP_PX:
                letter = 'Z'

        display_letter = letter if confidence >= CONFIDENCE_THRESHOLD else "?"
        cv2.putText(frame, f'Sign: {display_letter} ({confidence:.0%})',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # ── Dwell-timer logic ──────────────────────────────────────────────
        now = time.time()
        no_hand_since = None  # hand is present, reset no-hand timer

        if confidence >= CONFIDENCE_THRESHOLD and display_letter not in ('?', 'nothing'):
            # Reset dwell if the letter changed
            if letter != dwell_letter:
                dwell_letter = letter
                dwell_start  = now

            elapsed = now - dwell_start
            # Draw filling arc to show dwell progress
            frac    = min(elapsed / DWELL_TIME, 1.0)
            angle   = int(frac * 360)
            center  = (w - 55, 55)
            cv2.ellipse(frame, center, (30, 30), -90, 0, angle,
                        (0, 255, 180), 4)
            cv2.putText(frame, display_letter, (w - 67, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 180), 2)

            # Auto-add when dwell threshold reached
            cooldown_ok = (now - last_add_time) >= COOLDOWN_TIME
            if elapsed >= DWELL_TIME and cooldown_ok:
                if letter == 'space':
                    sentence += ' '
                elif letter == 'del':
                    sentence = sentence[:-1]
                else:
                    sentence += letter
                last_added_letter = letter
                last_add_time     = now
                dwell_start       = now   # restart dwell for repeated letters
                print(f"[AUTO] Added '{letter}' | Sentence: '{sentence}'")
        else:
            # Confidence dropped or no good letter — reset dwell
            dwell_letter = None
            dwell_start  = None

    else:
        prediction_buffer.clear()
        motion_history.clear()
        dwell_letter = None
        dwell_start  = None
        cv2.putText(frame, 'No hand detected',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # ── Auto-space when hand is absent long enough ─────────────────────
        now = time.time()
        if no_hand_since is None:
            no_hand_since = now
        elif (now - no_hand_since) >= NO_HAND_SPACE_T:
            if sentence and sentence[-1] != ' ':
                sentence += ' '
                last_add_time = now
                no_hand_since = now   # reset so it doesn't spam
                print(f"[AUTO] Space inserted | Sentence: '{sentence}'")

    # --- Draw UI ---
    cv2.putText(frame, f'Sentence: {sentence}',
                (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    if sentiment_display:
        color = (0, 255, 0) if "POSITIVE" in sentiment_display else \
                (0, 0, 255) if "NEGATIVE" in sentiment_display else (200, 200, 200)
        cv2.putText(frame, sentiment_display,
                    (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Dwell progress bar at bottom (only when tracking)
    if dwell_start is not None and confidence >= CONFIDENCE_THRESHOLD:
        frac = min((time.time() - dwell_start) / DWELL_TIME, 1.0)
        bar_w = int((w - 40) * frac)
        cv2.rectangle(frame, (20, h - 8), (w - 20, h - 3), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, h - 8), (20 + bar_w, h - 3), (0, 255, 180), -1)

    cv2.putText(frame, "A=Add(manual)  S=Sentiment  C=Clear  Q=Quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    cv2.imshow('ASL Sentiment Analyzer', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('a'):
        if hand_detected and confidence >= CONFIDENCE_THRESHOLD:
            if letter == 'space':
                sentence += ' '
            elif letter == 'del':
                sentence = sentence[:-1]
            elif letter != 'nothing':
                sentence += letter
            print(f"Current sentence: '{sentence}'")
        else:
            print(f"Low confidence or no hand detected, letter not added.")

    elif key == ord('s'):
        if sentence.strip():
            scores = analyzer.polarity_scores(sentence)
            if scores['compound'] >= 0.05:
                sentiment = "POSITIVE"
            elif scores['compound'] <= -0.05:
                sentiment = "NEGATIVE"
            else:
                sentiment = "NEUTRAL"
            sentiment_display = f"Sentiment: {sentiment}"
            print(f"Sentence: '{sentence}' | Sentiment: {sentiment} | Scores: {scores}")
        else:
            print("Sentence is empty, nothing to analyze.")

    elif key == ord('c'):
        sentence = ""
        sentiment_display = ""
        print("Sentence cleared.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()