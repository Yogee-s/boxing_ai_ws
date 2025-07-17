import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# ============================================================================
# SPATIAL-TEMPORAL GRAPH CONVOLUTIONAL NETWORK (ST-GCN)
# This is the BEST approach for skeleton-based action recognition
# ============================================================================

# Human skeleton adjacency matrix for 17 keypoints (COCO format)
def get_skeleton_adjacency():
    """Create adjacency matrix for human skeleton connections"""
    # COCO 17 keypoints connections
    edges = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
        (5, 11), (6, 12), (11, 12),  # torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # legs
    ]
    
    num_joints = 17
    A = np.zeros((num_joints, num_joints))
    
    # Add connections
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    
    # Add self-connections
    A += np.eye(num_joints)
    
    # Normalize
    D = np.sum(A, axis=1)
    A = A / D[:, np.newaxis]
    
    return A

class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.A = nn.Parameter(torch.FloatTensor(A), requires_grad=False)
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        # x: (N, C, T, V) where V=17 joints
        x = self.conv(x)  # (N, out_channels, T, V)
        x = torch.einsum('nctv,vw->nctw', x, self.A)  # Graph convolution
        return F.relu(self.bn(x))

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, (kernel_size, 1), 
                             (stride, 1), (pad, 0))
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1):
        super().__init__()
        self.gcn = GraphConv(in_channels, out_channels, A)
        self.tcn = TemporalConv(out_channels, out_channels, stride=stride)
        
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()
    
    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        return F.relu(x + res)

class STGCN(nn.Module):
    def __init__(self, num_classes=8, in_channels=2):  # 2 for (x,y) coordinates
        super().__init__()
        A = get_skeleton_adjacency()
        
        self.blocks = nn.ModuleList([
            STGCNBlock(in_channels, 64, A),
            STGCNBlock(64, 64, A),
            STGCNBlock(64, 64, A),
            STGCNBlock(64, 128, A, stride=2),
            STGCNBlock(128, 128, A),
            STGCNBlock(128, 128, A),
            STGCNBlock(128, 256, A, stride=2),
            STGCNBlock(256, 256, A),
            STGCNBlock(256, 256, A),
        ])
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x: (N, C, T, V) format
        for block in self.blocks:
            x = block(x)
        return self.classifier(x)

# ============================================================================
# ENHANCED DATASET FOR GRAPH FORMAT
# ============================================================================

class STGCNDataset(Dataset):
    def __init__(self, df, seq_len=50, step=10, augment=False):
        self.data = []
        self.seq_len = seq_len
        self.augment = augment
        
        for _, row in df.iterrows():
            arr = np.load(row.npy_path)  # (T, 34) -> (T, 17, 2)
            T = arr.shape[0]
            
            # Reshape to (T, 17, 2) for 17 joints with x,y coordinates
            arr = arr.reshape(T, 17, 2)
            
            if T < seq_len:
                # Pad short sequences
                pad = np.zeros((seq_len - T, 17, 2))
                arr = np.concatenate([arr, pad], axis=0)
                self.data.append((arr, row.class_idx))
            else:
                # Create overlapping windows
                for start in range(0, T - seq_len + 1, step):
                    window = arr[start:start+seq_len]
                    self.data.append((window, row.class_idx))
    
    def augment_pose(self, pose):
        """Advanced pose augmentation"""
        if not self.augment:
            return pose
        
        # Random rotation (around center)
        if np.random.rand() < 0.5:
            angle = np.random.uniform(-0.2, 0.2)  # ±0.2 radians
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            # Apply rotation to all joints
            center = pose.mean(axis=(0, 1))
            pose_centered = pose - center
            pose = np.dot(pose_centered, R.T) + center
        
        # Random scale
        if np.random.rand() < 0.3:
            scale = np.random.uniform(0.9, 1.1)
            center = pose.mean(axis=(0, 1))
            pose = (pose - center) * scale + center
        
        # Random translation
        if np.random.rand() < 0.3:
            tx = np.random.uniform(-0.05, 0.05)
            ty = np.random.uniform(-0.05, 0.05)
            pose[:, :, 0] += tx
            pose[:, :, 1] += ty
        
        # Add noise
        if np.random.rand() < 0.4:
            noise = np.random.normal(0, 0.01, pose.shape)
            pose += noise
        
        return pose
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        pose, label = self.data[idx]
        
        # Apply augmentation
        pose = self.augment_pose(pose.copy())
        
        # Normalize pose (important for graph networks)
        pose = self.normalize_pose(pose)
        
        # Convert to (C, T, V) format: (2, seq_len, 17)
        pose = torch.from_numpy(pose).permute(2, 0, 1).float()
        
        return pose, torch.tensor(label, dtype=torch.long)
    
    def normalize_pose(self, pose):
        """Normalize pose to be translation and scale invariant"""
        # Center around hip midpoint (joints 11, 12)
        hip_center = (pose[:, 11] + pose[:, 12]) / 2
        pose = pose - hip_center[:, np.newaxis, :]
        
        # Scale by torso length
        torso_length = np.linalg.norm(pose[:, 5] - pose[:, 11], axis=1).mean()
        if torso_length > 0:
            pose = pose / torso_length
        
        return pose

# ============================================================================
# TRAINING SETUP
# ============================================================================

# Configuration
LABELS_CSV = "dataset_with_poses/labels.csv"
CLASSES = ["block","cross","idle","jab","left_hook","left_uppercut","right_hook","right_uppercut"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
df = pd.read_csv(LABELS_CSV)
df['class_idx'] = df['label'].map({c:i for i,c in enumerate(CLASSES)})
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['class_idx'], random_state=42)

# Create datasets
train_ds = STGCNDataset(train_df, seq_len=50, step=10, augment=True)
val_ds = STGCNDataset(val_df, seq_len=50, step=10, augment=False)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

# Model
model = STGCN(num_classes=len(CLASSES)).to(DEVICE)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training setup
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return total_loss / len(loader), correct / total

def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    return total_loss / len(loader), correct / total, all_preds, all_targets

# Training loop
best_val_acc = 0
patience = 0
max_patience = 10

for epoch in range(1, 101):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc, preds, targets = val_epoch(model, val_loader, criterion, DEVICE)
    
    scheduler.step()
    
    print(f'Epoch {epoch:3d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
          f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_stgcn_boxing.pth')
        patience = 0
        print(f'✓ New best model saved! Val Acc: {val_acc:.4f}')
    else:
        patience += 1
        if patience >= max_patience:
            print('Early stopping!')
            break
    
    # Print detailed results every 20 epochs
    if epoch % 20 == 0:
        print('\nClassification Report:')
        print(classification_report(targets, preds, target_names=CLASSES, zero_division=0))
        print('Confusion Matrix:')
        print(confusion_matrix(targets, preds))
        print('-' * 80)

print(f'\nBest validation accuracy: {best_val_acc:.4f}')