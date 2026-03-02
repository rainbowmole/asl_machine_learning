# ASL Machine Learning

A two-phase machine learning project for recognising **American Sign Language (ASL)** gestures using hand landmark detection powered by [MediaPipe](https://developers.google.com/mediapipe) and [TensorFlow](https://www.tensorflow.org/).

---

## 📖 Overview

This project is split into two development phases:

- **Phase 1 (Prototype):** Static ASL alphabet letter recognition using image-based training data and hand landmark feature extraction.
- **Phase 2 (Landmark):** Dynamic ASL word/phrase recognition using real-time webcam sequences of hand landmarks.

---

## 🗂️ Project Structure

```
asl_machine_learning/
│
├── phase_1_asl(prototype)/
│   ├── extract_landmarks.py      # Extracts 95 hand landmark features from ASL alphabet images
│   ├── train_landmark_asl.py     # Trains a landmark-based ASL letter classifier
│   ├── machine-learning.py       # Transfer learning with MobileNetV2 on raw images
│   └── app.py                    # Real-time ASL letter recognition app with sentiment analysis
│
└── phase_2_landmark/
    ├── collect-data.py           # Collects webcam gesture sequences for training
    └── train_landmark.py         # Trains an LSTM-based model on gesture sequences
```

---

## 🚀 Phase 1 — ASL Alphabet Recognition (Prototype)

Phase 1 focuses on recognising static ASL alphabet letters from images and a live webcam feed.

### Scripts

| Script | Description |
|---|---|
| `extract_landmarks.py` | Processes the ASL alphabet image dataset and extracts 95 hand landmark features per image into a CSV file. |
| `train_landmark_asl.py` | Trains a dense neural network on the extracted landmark CSV. Includes mirror augmentation and noise augmentation. |
| `machine-learning.py` | Alternative image-based approach using MobileNetV2 transfer learning on raw ASL alphabet images. |
| `app.py` | Real-time webcam app that predicts ASL letters, auto-adds letters to a word, and performs VADER sentiment analysis on the composed text. |

### Feature Extraction (95 features)

- 63 scale-normalised landmark coordinates (wrist-relative x/y/z)
- 3 palm normal vector components
- 5 finger curl distances (fingertip-to-MCP)
- 5 PIP joint bend angles
- 5 DIP joint bend angles
- 10 pairwise fingertip distances
- 4 thumb-tip to fingertip distances

### App Controls

| Key | Action |
|---|---|
| `A` | Manually add the detected letter |
| `S` | Run VADER sentiment analysis on composed text |
| `C` | Clear the current text |
| `Q` | Quit the application |

> The app also supports **auto-mode**: hold a letter steady for 1.5 s to add it automatically, or remove your hand for 1.5 s to insert a space.

### Running Phase 1

```bash
cd phase_1_asl(prototype)

# Step 1: Extract landmarks from the image dataset
python extract_landmarks.py

# Step 2: Train the classifier
python train_landmark_asl.py

# Step 3: Launch the real-time app
python app.py
```

---

## 🚀 Phase 2 — Dynamic Gesture Recognition (Landmark Sequences)

Phase 2 extends the project to recognise **multi-word ASL signs** using sequences of hand landmarks captured over time.

### Supported Signs

`HAPPY`, `GOOD`, `LOVE`, `EXCITED`, `SAD`, `ANGRY`, `SCARED`, `HATE`, `OKAY`, `FEEL`

### Scripts

| Script | Description |
|---|---|
| `collect-data.py` | Captures 30-frame gesture sequences via webcam and saves them as `.npy` files per label. |
| `train_landmark.py` | Loads saved sequences, applies mirror augmentation (doubling the dataset), and trains an LSTM-based sequence model. |

### Frame Format

Each frame is a fixed **128-value vector**:
```
[left hand 63 values] + [right hand 63 values] + [left_present flag] + [right_present flag]
```

### Data Collection Controls

| Key | Action |
|---|---|
| `SPACE` | Start recording a sequence |
| `N` | Move to the next label |
| `B` | Go back to the previous label |
| `Q` | Quit |

### Running Phase 2

```bash
cd phase_2_landmark

# Step 1: Collect gesture data
python collect-data.py

# Step 2: Train the model
python train_landmark.py
```

---

## 🛠️ Requirements

- Python 3.8+
- [TensorFlow](https://www.tensorflow.org/)
- [MediaPipe](https://developers.google.com/mediapipe)
- [OpenCV](https://opencv.org/) (`cv2`)
- [NumPy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)
- [vaderSentiment](https://github.com/cjhutto/vaderSentiment) *(Phase 1 app only)*

Install dependencies:

```bash
pip install tensorflow mediapipe opencv-python numpy scikit-learn pandas vaderSentiment
```

> **Phase 2** also requires the `hand_landmarker.task` model file from MediaPipe, placed in the `phase_2_landmark/` directory. Download it from the [MediaPipe Models page](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models).

---

## ✨ Key Features

- 🤚 **Two-hand support** in Phase 2 — tracks both left and right hands simultaneously
- 🔁 **Mirror augmentation** — automatically doubles training data by generating mirrored gesture sequences, making the model hand-agnostic
- 📈 **Noise augmentation** — adds small random noise to training data to improve generalisation
- 💬 **Sentiment analysis** — Phase 1 app analyses the emotional tone of signed text using VADER
- ⚡ **Real-time inference** — both phases support live webcam prediction