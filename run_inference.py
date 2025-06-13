import cv2
import numpy as np
import torch
from collections import deque
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
POSE_MODEL_PATH   = "models/yolo11n-pose.pt"
CLS_MODEL_PATH    = "best_boxing_lstm.pth"
SEQ_LEN           = 50               # same as training
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# remake your class list in the same order you trained
CLASSES = ["block","cross","idle","jab","left_hook","left_uppercut","right_hook","right_uppercut"]

# COCO skeleton if you want to draw it
SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (0,5),(0,6),(5,7),(7,9),(6,8),(8,10),
    (5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)
]

# ──────────────────────────────────────────────────────────────────────────────
# Load models
# ──────────────────────────────────────────────────────────────────────────────
pose_model = YOLO(POSE_MODEL_PATH)
pose_model.fuse()  # speed up if supported

# define your LSTMClassifier (copy the exact class from training cell)
class LSTMClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=len(CLASSES), dp=0.3):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers,
                                  batch_first=True, dropout=dp)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim//2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dp),
            torch.nn.Linear(hidden_dim//2, num_classes)
        )
    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        last = h_n[-1]
        return self.head(last)

# instantiate and load weights
# infer input_dim from pose model (num_kpts*2)
# run one dummy inference to get keypoint count:
_ = pose_model(np.zeros((384,640,3), dtype=np.uint8))
num_kpts = _.keypoints.xy.shape[-2]
input_dim = num_kpts*2

cls_model = LSTMClassifier(input_dim).to(DEVICE)
cls_model.load_state_dict(torch.load(CLS_MODEL_PATH, map_location=DEVICE))
cls_model.eval()

# ──────────────────────────────────────────────────────────────────────────────
# Real‐time loop
# ──────────────────────────────────────────────────────────────────────────────
buffer = deque(maxlen=SEQ_LEN)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # pose detection expects RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose_model(rgb)[0]

    # extract first‐person keypoints or zeros
    if res.keypoints is not None and res.keypoints.xy.shape[0]>0:
        kps = res.keypoints.xy.cpu().numpy()[0]  # (num_kpts,2)
    else:
        kps = np.zeros((num_kpts,2), dtype=float)

    # flatten & append to buffer
    flat = kps.flatten()
    buffer.append(flat)

    # once we have a full sequence, classify
    if len(buffer) == SEQ_LEN:
        seq = np.stack(buffer, axis=0)                     # (SEQ_LEN, D)
        seq = (seq - seq.mean(0)) / (seq.std(0) + 1e-6)    # normalize
        x = torch.from_numpy(seq).unsqueeze(0).to(DEVICE).float()  # (1,SEQ_LEN,D)
        with torch.no_grad():
            logits = cls_model(x)
            pred = logits.argmax(1).item()
            label = CLASSES[pred]
            prob  = torch.softmax(logits,1)[0,pred].item()

        # overlay text
        cv2.putText(frame,
                    f"{label} ({prob*100:.1f}%)",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    # (optional) draw skeleton
    for (i,j) in SKELETON:
        x1,y1 = map(int, kps[i])
        x2,y2 = map(int, kps[j])
        cv2.line(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    for x,y in kps:
        cv2.circle(frame, (int(x),int(y)), 3, (0,0,255), -1)

    # show
    cv2.imshow("Boxing Move Predictor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
