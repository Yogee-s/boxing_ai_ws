#!/usr/bin/env python3
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from ultralytics import YOLO
import os

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
POSE_MODEL_PATH = "models/yolo11n-pose.pt"
CLS_MODEL_PATH  = "best_stgcn_boxing.pth"
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
num_kpts = 17

# ──────────────────────────────────────────────────────────────────────────────
# ST-GCN MODEL DEFINITION
# ──────────────────────────────────────────────────────────────────────────────

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, A):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.A = nn.Parameter(A.clone())
        self.conv = nn.Conv2d(in_features, out_features, 1)
        self.bn = nn.BatchNorm2d(out_features)
        
    def forward(self, x):
        # x: (N, C, T, V) where N=batch, C=channels, T=time, V=vertices
        N, C, T, V = x.size()
        
        # Apply adjacency matrix
        x = torch.einsum('nctv,vw->nctw', x, self.A)
        x = self.conv(x)
        x = self.bn(x)
        
        return x

class TemporalConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(TemporalConvolution, self).__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, (kernel_size, 1), 
                             stride=(stride, 1), padding=(pad, 0))
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(STGCNBlock, self).__init__()
        
        self.gcn = GraphConvolution(in_channels, out_channels, A)
        self.tcn = TemporalConvolution(out_channels, out_channels, stride=stride)
        
        self.relu = nn.ReLU(inplace=True)
        
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.relu(x)
        x = self.tcn(x)
        
        return self.relu(x + res)

class STGCN(nn.Module):
    def __init__(self, num_classes, num_joints=17, in_channels=2):
        super(STGCN, self).__init__()
        
        # Create adjacency matrix for human skeleton
        A = self.create_adjacency_matrix(num_joints)
        
        # ST-GCN blocks - all with residual connections to match saved model
        self.blocks = nn.ModuleList([
            STGCNBlock(in_channels, 64, A, residual=True),
            STGCNBlock(64, 64, A),
            STGCNBlock(64, 64, A),
            STGCNBlock(64, 128, A, stride=2, residual=True),
            STGCNBlock(128, 128, A),
            STGCNBlock(128, 128, A),
            STGCNBlock(128, 256, A, stride=2, residual=True),
            STGCNBlock(256, 256, A),
            STGCNBlock(256, 256, A),
        ])
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
        
    def create_adjacency_matrix(self, num_joints):
        # Create adjacency matrix based on skeleton connections
        A = np.zeros((num_joints, num_joints))
        
        # Add self-connections
        for i in range(num_joints):
            A[i, i] = 1
            
        # Add skeleton connections
        for i, j in SKELETON:
            A[i, j] = 1
            A[j, i] = 1
            
        # Normalize
        D = np.sum(A, axis=1)
        D = np.diag(1.0 / np.sqrt(D + 1e-6))
        A = D @ A @ D
        
        return torch.FloatTensor(A)
        
    def forward(self, x):
        # x: (N, C, T, V)
        for block in self.blocks:
            x = block(x)
            
        return self.classifier(x)

# ──────────────────────────────────────────────────────────────────────────────
# LOAD & PREPARE MODELS
# ──────────────────────────────────────────────────────────────────────────────

# Check if model files exist
if not os.path.exists(POSE_MODEL_PATH):
    print(f"ERROR: Pose model not found at {POSE_MODEL_PATH}")
    exit(1)

if not os.path.exists(CLS_MODEL_PATH):
    print(f"ERROR: Classification model not found at {CLS_MODEL_PATH}")
    exit(1)

# 1) Pose model
print("Loading pose model...")
pose_model = YOLO(POSE_MODEL_PATH)
pose_model.fuse()  # optimize for inference

# 2) ST-GCN classifier
print("Loading ST-GCN classification model...")
cls_model = STGCN(num_classes=len(CLASSES), num_joints=num_kpts, in_channels=2).to(DEVICE)

try:
    # Load the model state dict
    checkpoint = torch.load(CLS_MODEL_PATH, map_location=DEVICE)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        cls_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        cls_model.load_state_dict(checkpoint)
        
    cls_model.eval()
    print("Models loaded successfully!")
except Exception as e:
    print(f"ERROR loading classification model: {e}")
    exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def normalize_keypoints(kps, frame_width, frame_height):
    """Normalize keypoints to [0, 1] range"""
    kps_norm = kps.copy()
    kps_norm[:, 0] = kps_norm[:, 0] / frame_width   # x coordinates
    kps_norm[:, 1] = kps_norm[:, 1] / frame_height  # y coordinates
    return kps_norm

def is_person_visible(kps, confidence_threshold=0.3):
    """Check if enough keypoints are visible"""
    valid_points = np.sum((kps[:, 0] > 0) & (kps[:, 1] > 0))
    return valid_points >= 8  # At least 8 visible keypoints

def smooth_prediction(predictions, window_size=5):
    """Smooth predictions using a sliding window"""
    if len(predictions) < window_size:
        return predictions[-1] if predictions else 0
    
    recent_preds = predictions[-window_size:]
    # Return most common prediction in recent window
    return max(set(recent_preds), key=recent_preds.count)

def prepare_stgcn_input(buffer):
    """Convert buffer to ST-GCN input format"""
    # buffer contains flattened keypoints (seq_len, 34) -> (seq_len, 17, 2)
    seq_len = len(buffer)
    keypoints = np.array(buffer).reshape(seq_len, num_kpts, 2)
    
    # Transpose to (C, T, V) format: (2, seq_len, 17)
    keypoints = keypoints.transpose(2, 0, 1)  # (2, T, V)
    
    # Add batch dimension: (N, C, T, V)
    keypoints = keypoints[np.newaxis, ...]  # (1, 2, T, V)
    
    return keypoints

# ──────────────────────────────────────────────────────────────────────────────
# REAL-TIME INFERENCE LOOP
# ──────────────────────────────────────────────────────────────────────────────

buffer = deque(maxlen=SEQ_LEN)
prediction_history = deque(maxlen=10)
frame_count = 0

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    exit(1)

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Starting real-time inference... Press 'q' to quit")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_count += 1
        frame_height, frame_width = frame.shape[:2]

        # 1) Pose detection (BGR→RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run pose detection with confidence threshold
        try:
            results = pose_model(rgb, conf=0.3, verbose=False)
            
            # Initialize keypoints as zeros
            kps = np.zeros((num_kpts, 2), dtype=float)
            kps_raw = np.zeros((num_kpts, 2), dtype=float)
            
            # Check if we have valid results
            if len(results) > 0:
                result = results[0]
                
                # Extract keypoints if available
                if (hasattr(result, 'keypoints') and 
                    result.keypoints is not None and 
                    len(result.keypoints.xy) > 0 and
                    result.keypoints.xy[0].shape[0] > 0):
                    
                    # Get the first (most confident) person
                    kps_raw = result.keypoints.xy[0].cpu().numpy()  # (17, 2)
                    
                    # Normalize keypoints
                    kps = normalize_keypoints(kps_raw, frame_width, frame_height)
                    
                    # Check if person is visible enough
                    if not is_person_visible(kps_raw):
                        kps = np.zeros((num_kpts, 2), dtype=float)
                        kps_raw = np.zeros((num_kpts, 2), dtype=float)
                else:
                    # No keypoints detected
                    cv2.putText(frame, "No person detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            else:
                # No detection results
                cv2.putText(frame, "No person detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                           
        except Exception as e:
            print(f"Pose detection error: {e}")
            kps = np.zeros((num_kpts, 2), dtype=float)
            kps_raw = np.zeros((num_kpts, 2), dtype=float)
            cv2.putText(frame, "Pose detection error", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # 3) Append to buffer
        buffer.append(kps.flatten())

        # 4) Classify when buffer is full
        current_pred = "Warming up..."
        confidence = 0.0
        
        if len(buffer) == SEQ_LEN:
            try:
                # Prepare input for ST-GCN
                stgcn_input = prepare_stgcn_input(buffer)
                
                # Convert to tensor
                x = torch.from_numpy(stgcn_input).to(DEVICE).float()
                
                with torch.no_grad():
                    logits = cls_model(x)
                    probs = torch.softmax(logits, dim=1)
                    pred = logits.argmax(dim=1).item()
                    confidence = probs[0, pred].item()
                
                # Only update prediction if confidence is high enough
                if confidence > 0.4:  # Confidence threshold
                    prediction_history.append(pred)
                    smoothed_pred = smooth_prediction(list(prediction_history))
                    current_pred = CLASSES[smoothed_pred]
                else:
                    current_pred = "Uncertain"
                    
            except Exception as e:
                print(f"Prediction error: {e}")
                current_pred = "Error"
                confidence = 0.0

        # 5) Draw skeleton on original coordinates
        if (np.any(kps_raw > 0)):  # Only draw if we have valid keypoints
            # Draw skeleton connections
            for a, b in SKELETON:
                x1, y1 = kps_raw[a]
                x2, y2 = kps_raw[b]
                if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw keypoints
            for i, (x, y) in enumerate(kps_raw):
                if x > 0 and y > 0:
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)

        # 6) Display prediction and info
        pred_text = f"{current_pred}"
        if confidence > 0:
            pred_text += f" ({confidence*100:.1f}%)"
            
        cv2.putText(frame, pred_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Buffer status
        buffer_text = f"Buffer: {len(buffer)}/{SEQ_LEN}"
        cv2.putText(frame, buffer_text, (10, frame_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 7) Display frame
        cv2.imshow("Boxing Move Predictor", frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nInterrupted by user")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Cleanup complete")