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

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, 'asl_landmark_model.keras')
LABELS_PATH = os.path.join(BASE_DIR, 'label_classes.npy')
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
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

# --- Camera setup ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

# ── Monochrome UI Colors (BGR) ───────────────────────────────────────────────
CYAN      = (220, 220, 220)   # primary white
CYAN_DIM  = (130, 130, 130)   # mid gray
CYAN_DARK = ( 45,  45,  45)   # dark fill / inactive
ORANGE    = (180, 180, 180)   # neutral gray (was orange for negative)

def draw_jarvis_corners(img, x1, y1, x2, y2, color, length=22, thickness=2):
    """Corner-bracket box instead of a full rectangle."""
    for px, py, sx, sy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(img, (px, py), (px + sx*length, py), color, thickness)
        cv2.line(img, (px, py), (px, py + sy*length), color, thickness)

sentence = ""
sentiment_display = ""
letter = "?"
confidence = 0.0

BUFFER_SIZE = 20  # increased for more stable averaging
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

    # --- Color display; detection on original RGB for accuracy ---
    display = frame.copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    hand_detected = False
    UI_HEIGHT  = 140
    TOP_HEIGHT = 40

    # Corner decorations on video frame
    CL = 28
    for cx, cy, sx, sy in [(0,0,1,1),(w-1,0,-1,1),(0,h-1,1,-1),(w-1,h-1,-1,-1)]:
        cv2.line(display, (cx, cy), (cx + sx*CL, cy), CYAN, 2)
        cv2.line(display, (cx, cy), (cx, cy + sy*CL), CYAN, 2)

    if result.multi_hand_landmarks:
        hand_detected = True
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(
            display, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=CYAN, thickness=2, circle_radius=3),
            mp_draw.DrawingSpec(color=CYAN_DIM, thickness=1)
        )

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
        # Corner-bracket bounding box
        x_coords = [lm[i].x * w for i in range(21)]
        y_coords = [lm[i].y * h for i in range(21)]
        bx1 = max(0, int(min(x_coords)) - 30)
        by1 = max(0, int(min(y_coords)) - 30)
        bx2 = min(w, int(max(x_coords)) + 30)
        by2 = min(h, int(max(y_coords)) + 30)
        box_color = CYAN if confidence >= CONFIDENCE_THRESHOLD else CYAN_DIM
        draw_jarvis_corners(display, bx1, by1, bx2, by2, box_color)
        label = f'[ {display_letter} ]  {confidence:.0%}'
        cv2.putText(display, label, (bx1, max(by1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2, cv2.LINE_AA)

        # ── Dwell-timer logic ──────────────────────────────────────────────
        now = time.time()
        no_hand_since = None  # hand is present, reset no-hand timer

        if confidence >= CONFIDENCE_THRESHOLD and display_letter not in ('?', 'nothing'):
            # Reset dwell if the letter changed
            if letter != dwell_letter:
                dwell_letter = letter
                dwell_start  = now

            elapsed = now - dwell_start
            # Dwell arc — B&W palette
            frac    = min(elapsed / DWELL_TIME, 1.0)
            angle   = int(frac * 360)
            center  = (w - 55, 55)
            cv2.ellipse(display, center, (32, 32), -90, 0, 360, CYAN_DARK, 4)
            cv2.ellipse(display, center, (32, 32), -90, 0, angle, CYAN, 4)
            cv2.putText(display, display_letter, (w - 67, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, CYAN, 2, cv2.LINE_AA)

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
        confidence   = 0.0
        letter       = "?"
        cv2.putText(display, '[ NO HAND DETECTED ]',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN_DIM, 1, cv2.LINE_AA)

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

    # ── JARVIS Top Bar ───────────────────────────────────────────────────────
    top_bar = np.zeros((TOP_HEIGHT, w, 3), dtype=np.uint8)
    cv2.rectangle(top_bar, (0, 0), (w, TOP_HEIGHT), (8, 6, 2), -1)
    cv2.line(top_bar, (0, TOP_HEIGHT - 1), (w, TOP_HEIGHT - 1), CYAN_DIM, 1)
    cv2.line(top_bar, (0, 0), (260, 0), CYAN, 2)
    cv2.putText(top_bar, '[ ASL INTERFACE v2.0 ]', (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, CYAN, 2, cv2.LINE_AA)
    ts = time.strftime('%H:%M:%S')
    cv2.putText(top_bar, ts, (w - 95, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, CYAN_DIM, 1, cv2.LINE_AA)
    status_col = CYAN if hand_detected else CYAN_DARK
    status_txt = 'TRACKING' if hand_detected else 'SCANNING'
    cv2.circle(top_bar, (w - 145, 19), 5, status_col, -1)
    cv2.putText(top_bar, status_txt, (w - 135, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, status_col, 1, cv2.LINE_AA)

    # ── JARVIS Bottom Panel ──────────────────────────────────────────────────
    panel = np.zeros((UI_HEIGHT, w, 3), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (w, UI_HEIGHT), (8, 6, 2), -1)
    cv2.line(panel, (0, 0), (w, 0), CYAN, 2)
    cv2.line(panel, (0, 2), (w, 2), CYAN_DARK, 1)

    # Divider: right column for confidence + letter
    div_x = w - 230
    cv2.line(panel, (div_x, 10), (div_x, UI_HEIGHT - 10), CYAN_DARK, 1)

    # OUTPUT label + sentence
    cv2.putText(panel, 'OUTPUT', (12, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, CYAN_DIM, 1, cv2.LINE_AA)
    cv2.line(panel, (12, 21), (75, 21), CYAN_DARK, 1)
    sentence_show = (sentence[-55:] if len(sentence) > 55 else sentence) or '_'
    cv2.putText(panel, sentence_show, (12, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, CYAN, 2, cv2.LINE_AA)

    # Sentiment
    if sentiment_display:
        s_color = CYAN if 'POSITIVE' in sentiment_display else \
                  ORANGE if 'NEGATIVE' in sentiment_display else CYAN_DIM
        cv2.putText(panel, sentiment_display, (12, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, s_color, 1, cv2.LINE_AA)

    # Controls hint
    cv2.putText(panel, '[A] ADD   [S] ANALYSE   [C] CLEAR   [Q] QUIT',
                (12, UI_HEIGHT - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (50, 50, 12), 1, cv2.LINE_AA)

    # Dwell progress bar
    if dwell_start is not None and confidence >= CONFIDENCE_THRESHOLD:
        frac = min((time.time() - dwell_start) / DWELL_TIME, 1.0)
        dw = int((div_x - 24) * frac)
        cv2.rectangle(panel, (12, 95), (div_x - 12, 108), CYAN_DARK, -1)
        cv2.rectangle(panel, (12, 95), (12 + dw, 108), CYAN, -1)
        cv2.rectangle(panel, (12, 95), (div_x - 12, 108), CYAN_DIM, 1)
        cv2.putText(panel, 'DWELL', (12, 123),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, CYAN_DIM, 1, cv2.LINE_AA)

    # Right: segmented confidence meter
    cv2.putText(panel, 'CONFIDENCE', (div_x + 10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, CYAN_DIM, 1, cv2.LINE_AA)
    n_segs, seg_w, seg_h, seg_gap = 10, 16, 20, 4
    seg_ox, seg_oy = div_x + 10, 26
    filled_segs = int(n_segs * min(confidence, 1.0))
    for i in range(n_segs):
        sx = seg_ox + i * (seg_w + seg_gap)
        col = CYAN if i < filled_segs else CYAN_DARK
        cv2.rectangle(panel, (sx, seg_oy), (sx + seg_w, seg_oy + seg_h), col, -1)
    conf_col = CYAN if confidence >= CONFIDENCE_THRESHOLD else CYAN_DIM
    cv2.putText(panel, f'{confidence:.0%}', (div_x + 10, seg_oy + seg_h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_col, 2, cv2.LINE_AA)

    # Large letter display (right column)
    cv2.putText(panel, letter, (div_x + 140, 110),
                cv2.FONT_HERSHEY_DUPLEX, 2.8,
                CYAN if confidence >= CONFIDENCE_THRESHOLD else CYAN_DARK,
                3, cv2.LINE_AA)

    combined = np.vstack([top_bar, display, panel])
    cv2.imshow('ASL SENTINEL', combined)

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
            sentiment_display = f"Sentiment: {sentiment}  ({scores['compound']:+.2f})"
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