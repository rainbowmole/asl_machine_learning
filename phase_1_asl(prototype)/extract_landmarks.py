"""
Extract hand landmarks from the ASL alphabet training images.
Produces asl_landmarks.csv with 95 features + label:
  63  scale-normalised x/y/z coords (wrist-relative, / wrist-to-mid-MCP)
   3  palm normal vector              (palm facing direction)
   5  finger curl distances           (fingertip-to-MCP, normalised)
   5  PIP joint bend angles           (middle knuckle — E/M/N/S/T/A/X)
   5  DIP joint bend angles           (fingertip knuckle — X/V/K fine curl)
  10  pairwise fingertip distances    (all C(5,2) pairs — U/V/K/G/H spread)
   4  thumb-tip to fingertip dists    (thumb contact — S/E/T/A/M/N)

Run from: phase_1_asl(prototype)/
  python extract_landmarks.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import mediapipe as mp
import csv
import sys

DATASET_DIR = '../asl_alphabet_train/asl_alphabet_train'
OUTPUT_CSV  = 'asl_landmarks.csv'
SKIP_CLASSES = set()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# ── Landmark index constants ─────────────────────────────────────────────────
FINGER_TIPS = [4,  8, 12, 16, 20]          # thumb, index, middle, ring, pinky tips
FINGER_MCPS = [2,  5,  9, 13, 17]          # corresponding MCP/CMC joints
TIP_PAIRS   = [(4,8),(4,12),(4,16),(4,20),  # all C(5,2)=10 tip-to-tip pairs
               (8,12),(8,16),(8,20),
               (12,16),(12,20),(16,20)]
# (prev_joint, bend_joint, next_joint) for PIP then DIP of each finger
PIP_TRIPLETS = [(1,2,3),(5,6,7),(9,10,11),(13,14,15),(17,18,19)]
DIP_TRIPLETS = [(2,3,4),(6,7,8),(10,11,12),(14,15,16),(18,19,20)]

NUM_FEATURES = 95  # 63+3+5+5+5+10+4


def _joint_angle(raw, p, j, n):
    """Angle (radians) at joint j given neighbours p and n."""
    v1 = raw[p] - raw[j]
    v2 = raw[n] - raw[j]
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return float(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def extract_features(lm):
    """
    Given 21 MediaPipe landmark objects, return a 95-value np.float32 array.
    Returns None if landmarks are invalid.
    """
    wx, wy, wz = lm[0].x, lm[0].y, lm[0].z
    raw = np.array([(p.x-wx, p.y-wy, p.z-wz) for p in lm],
                   dtype=np.float32)              # (21, 3)
    scale = np.linalg.norm(raw[9]) + 1e-6

    scaled     = (raw / scale).flatten()           # 63
    normal     = np.cross(raw[5], raw[17])
    normal    /= np.linalg.norm(normal) + 1e-6     # 3
    curls      = [np.linalg.norm(raw[t]-raw[m])/scale
                  for t, m in zip(FINGER_TIPS, FINGER_MCPS)]  # 5
    pip_angles = [_joint_angle(raw, p, j, n) for p, j, n in PIP_TRIPLETS]  # 5
    dip_angles = [_joint_angle(raw, p, j, n) for p, j, n in DIP_TRIPLETS]  # 5
    pair_dists = [np.linalg.norm(raw[a]-raw[b])/scale for a, b in TIP_PAIRS] # 10
    thumb_dist = [np.linalg.norm(raw[4]-raw[t])/scale for t in [8,12,16,20]] # 4

    return np.concatenate(
        [scaled, normal, curls, pip_angles, dip_angles, pair_dists, thumb_dist]
    ).astype(np.float32)  # 95


def process_image(path):
    """Return 95-value feature list or None if no hand detected."""
    img = cv2.imread(path)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        return None
    return extract_features(result.multi_hand_landmarks[0].landmark)


# ── Main extraction loop ───────────────────────────────────────────────────────
classes = sorted(os.listdir(DATASET_DIR))
print(f"Found {len(classes)} classes: {classes}")

total_saved = 0
total_skipped = 0

with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    header = [f'f{i}' for i in range(NUM_FEATURES)] + ['label']
    writer.writerow(header)

    for cls in classes:
        if cls in SKIP_CLASSES:
            print(f"  Skipping class: {cls}")
            continue

        cls_dir = os.path.join(DATASET_DIR, cls)
        images = [fn for fn in os.listdir(cls_dir)
                  if fn.lower().endswith(('.jpg', '.jpeg', '.png'))]

        saved = 0
        skipped = 0
        for i, fn in enumerate(images):
            feats = process_image(os.path.join(cls_dir, fn))
            if feats is None:
                skipped += 1
                continue

            writer.writerow(list(feats) + [cls])
            saved += 1

            if (i + 1) % 500 == 0:
                print(f"  [{cls}] {i+1}/{len(images)} processed, {saved} saved...")
                sys.stdout.flush()

        print(f"  [{cls}] Done — {saved} saved, {skipped} skipped (no hand)")
        total_saved += saved
        total_skipped += skipped

hands.close()
print(f"\nFinished! Total saved: {total_saved} | Skipped: {total_skipped}")
print(f"Output: {OUTPUT_CSV}")
