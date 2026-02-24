import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import urllib.request
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

MODEL_PATH = 'asl_model.h5'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

dummy = np.zeros((1, 64, 64, 3))
model(dummy, training=False)
print("Model ready! Controls: [A] Add letter | [S] Analyze sentiment | [C] Clear | [Q] Quit")

labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['del', 'nothing', 'space']
analyzer = SentimentIntensityAnalyzer()

# --- MediaPipe hand tracking setup (new Tasks API) ---
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        model_path
    )
    print("Model downloaded!")

base_options = mp_python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

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
CONFIDENCE_THRESHOLD = 0.85

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # --- MediaPipe hand detection (new Tasks API) ---
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_image)

    hand_detected = False

    if result.hand_landmarks:
        hand_detected = True
        hand_landmarks = result.hand_landmarks[0]

        x_coords = [lm.x * w for lm in hand_landmarks]
        y_coords = [lm.y * h for lm in hand_landmarks]

        x1, x2 = int(min(x_coords)) - 40, int(max(x_coords)) + 40
        y1, y2 = int(min(y_coords)) - 40, int(max(y_coords)) + 40
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Draw landmarks manually
        for lm in hand_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
            img = cv2.resize(roi_gray, (64, 64))
            img = np.expand_dims(img / 255.0, axis=0)

            prediction = model(img, training=False).numpy()
            raw_letter = labels[np.argmax(prediction)]
            confidence = np.max(prediction)

            prediction_buffer.append(raw_letter)
            if len(prediction_buffer) > BUFFER_SIZE:
                prediction_buffer.pop(0)
            letter = Counter(prediction_buffer).most_common(1)[0][0]

            display_letter = letter if confidence >= CONFIDENCE_THRESHOLD else "?"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Sign: {display_letter} ({confidence:.0%})',
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        prediction_buffer.clear()
        cv2.putText(frame, 'No hand detected',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # --- Draw UI ---
    cv2.putText(frame, f'Sentence: {sentence}',
                (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    if sentiment_display:
        color = (0, 255, 0) if "POSITIVE" in sentiment_display else \
                (0, 0, 255) if "NEGATIVE" in sentiment_display else (200, 200, 200)
        cv2.putText(frame, sentiment_display,
                    (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(frame, "A=Add  S=Sentiment  C=Clear  Q=Quit",
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