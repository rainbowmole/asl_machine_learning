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
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")
if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"Labels file '{LABELS_PATH}' not found.")

model  = tf.keras.models.load_model(MODEL_PATH, compile=False)
labels = list(np.load(LABELS_PATH, allow_pickle=True))

# --- Compile + warmup ---
@tf.function(reduce_retracing=True)
def predict(x):
    return model(x, training=False)

dummy = np.zeros((1, 95), dtype=np.float32)
predict(dummy)

print(f"Model ready! {len(labels)} classes: {labels}")
print("Controls: [A] Add manual | [S] Sentiment | [C] Clear | [Q] Quit")
print("Auto-mode: hold steady for 1.5s | Remove hand 1.5s = space")

analyzer = SentimentIntensityAnalyzer()

# --- Landmark constants ---
_TIPS         = [4,  8, 12, 16, 20]
_MCPS         = [2,  5,  9, 13, 17]
_TIP_PAIRS    = [(4,8),(4,12),(4,16),(4,20),
                 (8,12),(8,16),(8,20),
                 (12,16),(12,20),(16,20)]
_PIP_TRIPLETS = [(1,2,3),(5,6,7),(9,10,11),(13,14,15),(17,18,19)]
_DIP_TRIPLETS = [(2,3,4),(6,7,8),(10,11,12),(14,15,16),(18,19,20)]

def _joint_angle(raw, p, j, n):
    v1 = raw[p] - raw[j]
    v2 = raw[n] - raw[j]
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return float(np.arccos(np.clip(cos_a, -1.0, 1.0)))

def extract_features(lm):
    wx, wy, wz = lm[0].x, lm[0].y, lm[0].z
    raw        = np.array([(p.x-wx, p.y-wy, p.z-wz) for p in lm], dtype=np.float32)
    scale      = np.linalg.norm(raw[9]) + 1e-6
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
    ).astype(np.float32)

def draw_jarvis_corners(img, x1, y1, x2, y2, color, length=22, thickness=2):
    for px, py, sx, sy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(img, (px, py), (px + sx*length, py), color, thickness)
        cv2.line(img, (px, py), (px, py + sy*length), color, thickness)

def discriminate_nm(letter, raw, scale):
    """
    Geometric N vs M disambiguation.

    M (ASL): thumb tucked under index + middle + ring  → all three tips
             are similarly close to the thumb tip.
    N (ASL): thumb tucked under index + middle only    → ring tip is
             noticeably further from the thumb than the middle tip.

    Two independent cues are combined so partial heuristics still fire:
      1. Absolute distance: ring_to_thumb < 0.45  (was 0.35 – too tight)
      2. Ratio guard:       ring / middle  < 1.45  (M keeps them proportional)
    Both are required to avoid over-predicting M.
    """
    if letter not in ('N', 'M'):
        return letter
    middle_to_thumb = np.linalg.norm(raw[12] - raw[4]) / scale
    ring_to_thumb   = np.linalg.norm(raw[16] - raw[4]) / scale
    ring_mid_ratio  = ring_to_thumb / (middle_to_thumb + 1e-6)
    # M: ring is close AND proportionally similar to middle
    return 'M' if (ring_to_thumb < 0.45 and ring_mid_ratio < 1.45) else 'N'

# --- MediaPipe ---
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

# --- Camera ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

# --- B&W UI Colors ---
CYAN      = (220, 220, 220)
CYAN_DIM  = (130, 130, 130)
CYAN_DARK = ( 45,  45,  45)
ORANGE    = (180, 180, 180)

# --- State ---
sentence          = ""
sentiment_display = ""
letter            = "?"
confidence        = 0.0
frame_count       = 0

# --- Tuned constants ---
BUFFER_SIZE          = 8
CONFIDENCE_THRESHOLD = 0.82
DWELL_TIME           = 0.8
COOLDOWN_TIME        = 0.9
NO_HAND_SPACE_T      = 1.5     # kept for reference but auto-space is disabled
SPACE_CONFIDENCE_MIN = 0.96   # space sign needs very high confidence to dwell-commit

prediction_buffer = []
dwell_letter      = None
dwell_start       = None
last_added_letter = None
last_add_time     = 0.0
no_hand_since     = None

# --- Motion detection ---
MOTION_FRAMES    = 25       # frames of history (~0.8s at 30fps, every 2nd frame)
motion_history   = deque(maxlen=MOTION_FRAMES)
J_DROP_PX        = 28       # min net downward pixel travel  (J hook)
J_DOWN_FRAC      = 0.55     # fraction of frames that must go downward
J_MIN_PEAK_PX    = 5        # fastest single-frame drop must exceed this (rules out idle drift)
Z_SWEEP_PX       = 55       # min total horizontal sweep     (Z stroke)
Z_MIN_REVERSALS  = 2        # min direction reversals in X   (Z zig-zag — needs at least zig+zag)
Z_JITTER_PX      = 4        # ignore X deltas smaller than this (noise filter)
Z_MIN_PEAK_PX    = 6        # fastest single-frame X move must exceed this (rules out idle drift)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame_count += 1
    skip_prediction = (frame_count % 2 != 0)

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    display = frame.copy()
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result  = hands.process(rgb)

    hand_detected = False
    UI_HEIGHT     = 140
    TOP_HEIGHT    = 40

    # Corner decorations
    CL = 28
    for cx, cy, sx, sy in [(0,0,1,1),(w-1,0,-1,1),(0,h-1,1,-1),(w-1,h-1,-1,-1)]:
        cv2.line(display, (cx, cy), (cx + sx*CL, cy), CYAN, 2)
        cv2.line(display, (cx, cy), (cx, cy + sy*CL), CYAN, 2)

    if result.multi_hand_landmarks:
        hand_detected  = True
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(
            display, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=CYAN, thickness=2, circle_radius=3),
            mp_draw.DrawingSpec(color=CYAN_DIM, thickness=1)
        )

        lm = hand_landmarks.landmark

        # Predict every 2nd frame
        if not skip_prediction:
            feat       = extract_features(lm).reshape(1, -1)
            prediction = predict(feat).numpy()[0]
            prediction_buffer.append(prediction)
            if len(prediction_buffer) > BUFFER_SIZE:
                prediction_buffer.pop(0)

        # Weighted average
        if prediction_buffer:
            weights    = np.linspace(0.5, 1.0, len(prediction_buffer))
            weights   /= weights.sum()
            avg_probs  = np.average(prediction_buffer, axis=0, weights=weights)
            letter     = labels[np.argmax(avg_probs)]
            confidence = np.max(avg_probs)

        # Raw landmarks for geometric checks
        wx, wy, wz = lm[0].x, lm[0].y, lm[0].z
        raw_lm     = np.array(
            [(p.x-wx, p.y-wy, p.z-wz) for p in lm], dtype=np.float32
        )
        scale = np.linalg.norm(raw_lm[9]) + 1e-6

        # J / Z motion detection
        pinky_y_px = lm[20].y * h
        index_x_px = lm[8].x  * w
        motion_history.append((pinky_y_px, index_x_px))

        if len(motion_history) == MOTION_FRAMES:
            pinky_ys = [m[0] for m in motion_history]
            index_xs = [m[1] for m in motion_history]

            motion_letter = None

            # --- J: net downward drop + consistently downward + fast peak ---
            # Peak velocity check: idle hand drift is all tiny deltas;
            # a real J stroke has at least one frame with a big drop.
            net_drop  = pinky_ys[-1] - pinky_ys[0]
            y_deltas  = [pinky_ys[i+1] - pinky_ys[i] for i in range(len(pinky_ys)-1)]
            down_frac = sum(1 for d in y_deltas if d > 0) / len(y_deltas)
            peak_drop = max(y_deltas)   # biggest single-frame downward movement
            if (letter == 'I'
                    and net_drop   > J_DROP_PX
                    and down_frac  > J_DOWN_FRAC
                    and peak_drop  > J_MIN_PEAK_PX):
                motion_letter = 'J'

            # --- Z: sweep + zig-zag reversals + fast peak ---
            else:
                x_sweep  = max(index_xs) - min(index_xs)
                x_deltas = [index_xs[i+1] - index_xs[i] for i in range(len(index_xs)-1)]
                solid    = [d for d in x_deltas if abs(d) > Z_JITTER_PX]
                reversals = sum(
                    1 for i in range(len(solid)-1) if solid[i] * solid[i+1] < 0
                )
                peak_sweep = max((abs(d) for d in x_deltas), default=0)
                if (letter in ('G','D','X','I')
                        and x_sweep    > Z_SWEEP_PX
                        and reversals  >= Z_MIN_REVERSALS
                        and peak_sweep > Z_MIN_PEAK_PX):
                    motion_letter = 'Z'

            # --- Auto-commit J/Z immediately on motion, no dwell needed ---
            if motion_letter:
                letter = motion_letter
                now_m  = time.time()
                if (now_m - last_add_time) >= COOLDOWN_TIME:
                    sentence      += motion_letter
                    last_add_time  = now_m
                    last_added_letter = motion_letter
                    dwell_letter   = None
                    dwell_start    = None
                    motion_history.clear()   # prevent re-trigger on same stroke
                    print(f"[MOTION] '{motion_letter}' → '{sentence}'")

        display_letter = letter if confidence >= CONFIDENCE_THRESHOLD else "?"

        # Corner bracket bounding box
        x_coords = [lm[i].x * w for i in range(21)]
        y_coords = [lm[i].y * h for i in range(21)]
        bx1 = max(0, int(min(x_coords)) - 30)
        by1 = max(0, int(min(y_coords)) - 30)
        bx2 = min(w, int(max(x_coords)) + 30)
        by2 = min(h, int(max(y_coords)) + 30)
        box_color = CYAN if confidence >= CONFIDENCE_THRESHOLD else CYAN_DIM
        draw_jarvis_corners(display, bx1, by1, bx2, by2, box_color)
        cv2.putText(display, f'[ {display_letter} ]  {confidence:.0%}',
                    (bx1, max(by1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2, cv2.LINE_AA)

        # Dwell logic
        now           = time.time()
        no_hand_since = None

        if confidence >= CONFIDENCE_THRESHOLD and display_letter not in ('?', 'nothing'):
            if letter != dwell_letter:
                dwell_letter = letter
                dwell_start  = now

            elapsed = now - dwell_start
            frac    = min(elapsed / DWELL_TIME, 1.0)
            angle   = int(frac * 360)
            center  = (w - 55, 55)
            cv2.ellipse(display, center, (32, 32), -90, 0, 360, CYAN_DARK, 4)
            cv2.ellipse(display, center, (32, 32), -90, 0, angle, CYAN, 4)
            cv2.putText(display, display_letter, (w - 67, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, CYAN, 2, cv2.LINE_AA)

            cooldown_ok        = (now - last_add_time) >= COOLDOWN_TIME
            space_confidence_ok = (letter != 'space') or (confidence >= SPACE_CONFIDENCE_MIN)
            if elapsed >= DWELL_TIME and cooldown_ok and space_confidence_ok:
                if letter == 'space':
                    sentence += ' '
                elif letter == 'del':
                    sentence = sentence[:-1]
                else:
                    sentence += letter
                last_added_letter = letter
                last_add_time     = now
                dwell_start       = now
                print(f"[AUTO] '{letter}' → '{sentence}'")
        else:
            dwell_letter = None
            dwell_start  = None

    else:
        prediction_buffer.clear()
        motion_history.clear()
        dwell_letter  = None
        dwell_start   = None
        confidence    = 0.0
        letter        = "?"
        cv2.putText(display, '[ NO HAND DETECTED ]',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN_DIM, 1, cv2.LINE_AA)

        # Auto-space on hand removal is disabled — use [A] or the space ASL pose.
        no_hand_since = None

    # --- Top bar ---
    top_bar = np.zeros((TOP_HEIGHT, w, 3), dtype=np.uint8)
    cv2.rectangle(top_bar, (0, 0), (w, TOP_HEIGHT), (8, 6, 2), -1)
    cv2.line(top_bar, (0, TOP_HEIGHT-1), (w, TOP_HEIGHT-1), CYAN_DIM, 1)
    cv2.line(top_bar, (0, 0), (260, 0), CYAN, 2)
    cv2.putText(top_bar, '[ ASL SENTINEL v2.0 ]', (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, CYAN, 2, cv2.LINE_AA)
    ts = time.strftime('%H:%M:%S')
    cv2.putText(top_bar, ts, (w - 95, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, CYAN_DIM, 1, cv2.LINE_AA)
    status_col = CYAN if hand_detected else CYAN_DARK
    status_txt = 'TRACKING' if hand_detected else 'SCANNING'
    cv2.circle(top_bar, (w - 145, 19), 5, status_col, -1)
    cv2.putText(top_bar, status_txt, (w - 135, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, status_col, 1, cv2.LINE_AA)
    cv2.putText(top_bar, f'{cap.get(cv2.CAP_PROP_FPS):.0f} FPS',
                (w - 210, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.38, CYAN_DIM, 1, cv2.LINE_AA)

    # --- Bottom panel ---
    panel = np.zeros((UI_HEIGHT, w, 3), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (w, UI_HEIGHT), (8, 6, 2), -1)
    cv2.line(panel, (0, 0), (w, 0), CYAN, 2)
    cv2.line(panel, (0, 2), (w, 2), CYAN_DARK, 1)

    div_x = w - 230
    cv2.line(panel, (div_x, 10), (div_x, UI_HEIGHT-10), CYAN_DARK, 1)

    # Output label + sentence
    cv2.putText(panel, 'OUTPUT', (12, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, CYAN_DIM, 1, cv2.LINE_AA)
    cv2.line(panel, (12, 21), (75, 21), CYAN_DARK, 1)
    sentence_show = ('...' + sentence[-52:]) if len(sentence) > 55 else (sentence or '_')
    cv2.putText(panel, sentence_show, (12, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, CYAN, 2, cv2.LINE_AA)

    # Sentiment
    if sentiment_display:
        s_color = CYAN   if 'POSITIVE' in sentiment_display else \
                  ORANGE if 'NEGATIVE' in sentiment_display else CYAN_DIM
        cv2.putText(panel, sentiment_display, (12, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, s_color, 1, cv2.LINE_AA)

    # Controls
    cv2.putText(panel, '[A] ADD   [S] ANALYSE   [C] CLEAR   [Q] QUIT',
                (12, UI_HEIGHT - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (50, 50, 12), 1, cv2.LINE_AA)

    # Dwell progress bar
    if dwell_start is not None and confidence >= CONFIDENCE_THRESHOLD:
        frac = min((time.time() - dwell_start) / DWELL_TIME, 1.0)
        dw   = int((div_x - 24) * frac)
        cv2.rectangle(panel, (12, 95), (div_x-12, 108), CYAN_DARK, -1)
        cv2.rectangle(panel, (12, 95), (12 + dw,  108), CYAN,      -1)
        cv2.rectangle(panel, (12, 95), (div_x-12, 108), CYAN_DIM,   1)
        cv2.putText(panel, 'DWELL', (12, 123),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, CYAN_DIM, 1, cv2.LINE_AA)

    # Confidence meter
    cv2.putText(panel, 'CONFIDENCE', (div_x+10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, CYAN_DIM, 1, cv2.LINE_AA)
    n_segs, seg_w, seg_h, seg_gap = 10, 16, 20, 4
    seg_ox, seg_oy = div_x + 10, 26
    filled_segs    = int(n_segs * min(confidence, 1.0))
    for i in range(n_segs):
        sx  = seg_ox + i * (seg_w + seg_gap)
        col = CYAN if i < filled_segs else CYAN_DARK
        cv2.rectangle(panel, (sx, seg_oy), (sx+seg_w, seg_oy+seg_h), col, -1)
    conf_col = CYAN if confidence >= CONFIDENCE_THRESHOLD else CYAN_DIM
    cv2.putText(panel, f'{confidence:.0%}', (div_x+10, seg_oy+seg_h+18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_col, 2, cv2.LINE_AA)

    # Large letter
    cv2.putText(panel, letter, (div_x+140, 110),
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
            print(f"[MANUAL] '{letter}' → '{sentence}'")
        else:
            print("Low confidence or no hand detected.")

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
            print(f"'{sentence}' → {sentiment} {scores}")
        else:
            print("Sentence is empty.")

    elif key == ord('c'):
        sentence          = ""
        sentiment_display = ""
        print("Cleared.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()