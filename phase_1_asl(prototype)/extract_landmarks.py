"""
Extract hand landmarks from the ASL alphabet training images.
Produces asl_landmarks.csv with 95 features + label.

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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

DATASET_DIR     = '../asl_alphabet_train/asl_alphabet_train'
OUTPUT_CSV      = 'asl_landmarks.csv'
SKIP_CLASSES    = {'nothing'}
MAX_WORKERS     = 4
MIN_DETECT_CONF = 0.7

FINGER_TIPS  = [4,  8, 12, 16, 20]
FINGER_MCPS  = [2,  5,  9, 13, 17]
TIP_PAIRS    = [(4,8),(4,12),(4,16),(4,20),
                (8,12),(8,16),(8,20),
                (12,16),(12,20),(16,20)]
PIP_TRIPLETS = [(1,2,3),(5,6,7),(9,10,11),(13,14,15),(17,18,19)]
DIP_TRIPLETS = [(2,3,4),(6,7,8),(10,11,12),(14,15,16),(18,19,20)]
NUM_FEATURES = 95

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
    curls      = [np.linalg.norm(raw[t]-raw[m])/scale
                  for t, m in zip(FINGER_TIPS, FINGER_MCPS)]
    pip_angles = [_joint_angle(raw, p, j, n) for p, j, n in PIP_TRIPLETS]
    dip_angles = [_joint_angle(raw, p, j, n) for p, j, n in DIP_TRIPLETS]
    pair_dists = [np.linalg.norm(raw[a]-raw[b])/scale for a, b in TIP_PAIRS]
    thumb_dist = [np.linalg.norm(raw[4]-raw[t])/scale for t in [8,12,16,20]]
    return np.concatenate(
        [scaled, normal, curls, pip_angles, dip_angles, pair_dists, thumb_dist]
    ).astype(np.float32)

def process_image(args):
    path, label = args
    mp_h = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=MIN_DETECT_CONF
    )
    try:
        img = cv2.imread(path)
        if img is None:
            return None
        rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = mp_h.process(rgb)
        if not result.multi_hand_landmarks:
            return None
        feats = extract_features(result.multi_hand_landmarks[0].landmark)
        return (feats, label)
    except Exception:
        return None
    finally:
        mp_h.close()

def format_eta(seconds):
    if seconds < 60:    return f"{int(seconds)}s"
    elif seconds < 3600: return f"{int(seconds//60)}m {int(seconds%60)}s"
    else:               return f"{int(seconds//3600)}h {int((seconds%3600)//60)}m"

# --- Resume check ---
processed_classes = set()
if os.path.exists(OUTPUT_CSV):
    print(f"Found existing {OUTPUT_CSV} — checking for resume...")
    with open(OUTPUT_CSV, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row:
                processed_classes.add(row[-1])
    if processed_classes:
        print(f"Already processed: {sorted(processed_classes)}")

classes    = sorted(os.listdir(DATASET_DIR))
write_mode = 'a' if processed_classes else 'w'

total_saved   = 0
total_skipped = 0
start_time    = time.time()

print(f"\nFound {len(classes)} classes: {classes}")

with open(OUTPUT_CSV, write_mode, newline='') as f:
    writer = csv.writer(f)

    if not processed_classes:
        header = [f'f{i}' for i in range(NUM_FEATURES)] + ['label']
        writer.writerow(header)

    for cls in classes:
        if cls in SKIP_CLASSES:
            print(f"  Skipping class: {cls}")
            continue
        if cls in processed_classes:
            print(f"  [{cls}] Already processed — skipping")
            continue

        cls_dir = os.path.join(DATASET_DIR, cls)
        images  = [fn for fn in os.listdir(cls_dir)
                   if fn.lower().endswith(('.jpg','.jpeg','.png'))]
        tasks     = [(os.path.join(cls_dir, fn), cls) for fn in images]
        saved     = 0
        skipped   = 0
        cls_start = time.time()

        print(f"\n  [{cls}] Processing {len(images)} images ({MAX_WORKERS} threads)...")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_image, t): t for t in tasks}
            done    = 0
            for future in as_completed(futures):
                result = future.result()
                done  += 1
                if result is not None:
                    feats, label = result
                    writer.writerow(list(feats) + [label])
                    saved += 1
                else:
                    skipped += 1
                if done % 500 == 0:
                    elapsed   = time.time() - cls_start
                    rate      = done / elapsed if elapsed > 0 else 1
                    remaining = (len(images) - done) / rate
                    print(f"    {done}/{len(images)} ({done/len(images)*100:.0f}%)"
                          f" | {rate:.0f} img/s | ETA: {format_eta(remaining)}")
                    sys.stdout.flush()

        cls_elapsed = time.time() - cls_start
        print(f"  [{cls}] Done — {saved} saved, {skipped} skipped "
              f"({cls_elapsed:.1f}s)")
        total_saved   += saved
        total_skipped += skipped

total_elapsed = time.time() - start_time
print(f"\n{'='*50}")
print(f"Finished! Saved: {total_saved} | Skipped: {total_skipped}")
print(f"Time: {format_eta(total_elapsed)} | Output: {OUTPUT_CSV}")