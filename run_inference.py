#!/usr/bin/env python3
import cv2
import numpy as np
import torch
from collections import deque
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
POSE_MODEL_PATH = "models/yolo11n-pose.pt"
CLS_MODEL_PATH  = "best_boxing_lstm.pth"
SEQ_LEN         = 50
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Your classes in the same order as training
CLASSES = [
    "block",
    "cross",
    "idle",
    "jab",
    "left_hook",
    "left_uppercut",
    "right_hook",
    "right_uppercut",
]

# COCO skeleton connections (0-indexed keypoints)
SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (0,5),(0,6),(5,7),(7,9),(6,8),(8,10),
    (5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)
]

# Hard-code number of keypoints to 17 (COCO)
num_kpts  = 17
input_dim = num_kpts * 2

# ──────────────────────────────────────────────────────────────────────────────
# LOAD & PREPARE MODELS
# ──────────────────────────────────────────────────────────────────────────────
# 1) Pose model
pose_model = YOLO(POSE_MODEL_PATH)
pose_model.fuse()  # optimize for inference

# 2) LSTM classifier (must match your training code)
class LSTMClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=len(CLASSES), dp=0.3):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim,
                                  hidden_dim,
                                  num_layers,
                                  batch_first=True,
                                  dropout=dp)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim//2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dp),
            torch.nn.Linear(hidden_dim//2, num_classes)
        )

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        last = h_n[-1]            # (batch, hidden_dim)
        return self.head(last)    # (batch, num_classes)

# Instantiate & load weights
cls_model = LSTMClassifier(input_dim).to(DEVICE)
cls_model.load_state_dict(torch.load(CLS_MODEL_PATH, map_location=DEVICE))
cls_model.eval()

# ──────────────────────────────────────────────────────────────────────────────
# REAL-TIME INFERENCE LOOP
# ──────────────────────────────────────────────────────────────────────────────
buffer = deque(maxlen=SEQ_LEN)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1) Pose detection (BGR→RGB)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose_model(rgb)[0]  # single Results object

    # 2) Extract keypoints
    if result.keypoints is not None and result.keypoints.xy.shape[0] > 0:
        kps = result.keypoints.xy.cpu().numpy()[0]  # (17,2)
    else:
        kps = np.zeros((num_kpts, 2), dtype=float)

    # 3) Append to buffer
    buffer.append(kps.flatten())

    # 4) Classify when buffer full
    if len(buffer) == SEQ_LEN:
        seq = np.stack(buffer, axis=0)                 # (SEQ_LEN, input_dim)
        seq = (seq - seq.mean(axis=0)) / (seq.std(axis=0) + 1e-6)
        x = torch.from_numpy(seq).unsqueeze(0).to(DEVICE).float()
        with torch.no_grad():
            logits = cls_model(x)
            pred  = logits.argmax(dim=1).item()
            conf  = torch.softmax(logits, dim=1)[0, pred].item()
        label = CLASSES[pred]
        cv2.putText(frame,
                    f"{label} ({conf*100:.1f}%)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA)

    # 5) Draw skeleton
    for a, b in SKELETON:
        x1, y1 = kps[a]
        x2, y2 = kps[b]
        if x1>0 and y1>0 and x2>0 and y2>0:
            cv2.line(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
    for x, y in kps:
        if x>0 and y>0:
            cv2.circle(frame, (int(x),int(y)), 3, (0,0,255), -1)

    # 6) Display & exit
    cv2.imshow("Boxing Move Predictor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
