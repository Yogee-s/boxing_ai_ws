import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
LABELS_CSV = "dataset_with_poses/labels.csv"
SEQ_LEN    = 50
STEP       = 10
BATCH_SIZE = 16
LR         = 1e-3
EPOCHS     = 50
PATIENCE   = 5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class labels in training order
CLASSES = ["block","cross","idle","jab","left_hook","left_uppercut","right_hook","right_uppercut"]

# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(LABELS_CSV)
df['class_idx'] = df['label'].map({c:i for i,c in enumerate(CLASSES)})
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['class_idx'], random_state=42)

class PoseWindowDataset(Dataset):
    def __init__(self, df, seq_len=SEQ_LEN, step=STEP):
        self.data = []
        for _, row in df.iterrows():
            arr = np.load(row.npy_path)  # (T, D)
            T, D = arr.shape
            if T < seq_len:
                pad = np.zeros((seq_len - T, D), dtype=arr.dtype)
                self.data.append((np.vstack([arr, pad]), row.class_idx))
            else:
                for start in range(0, T - seq_len + 1, step):
                    window = arr[start:start+seq_len]
                    self.data.append((window, row.class_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        arr, label = self.data[i]
        arr = (arr - arr.mean(axis=0)) / (arr.std(axis=0) + 1e-6)
        return torch.from_numpy(arr).float(), torch.tensor(label)

train_ds = PoseWindowDataset(train_df)
val_ds   = PoseWindowDataset(val_df)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
print(f"Train windows: {len(train_ds)}, Val windows: {len(val_ds)}")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITION
# ─────────────────────────────────────────────────────────────────────────────
# input_dim = dimension of each window frame = 2*num_kpts (load from one sample)
sample_x, _ = next(iter(train_loader))
input_dim = sample_x.shape[2]

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=len(CLASSES), dp=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dp)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dp),
            nn.Linear(hidden_dim//2, num_classes)
        )
    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        last = h_n[-1]
        return self.head(last)

model = LSTMClassifier(input_dim).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP WITH EARLY STOPPING & CONFUSION
# ─────────────────────────────────────────────────────────────────────────────
best_val_acc = 0.0
wait = 0
for epoch in range(1, EPOCHS+1):
    # train
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits = model(X)
        loss = criterion(logits, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        running_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    train_loss = running_loss / total
    train_acc  = correct / total

    # validate
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    val_acc = correct / total

    # confusion
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASSES))))

    print(f"Epoch {epoch:02d} | "
          f"Train loss: {train_loss:.3f}, train acc: {train_acc:.3f} | "
          f"Val acc: {val_acc:.3f}\nConfusion matrix:\n{cm}")

    # checkpoint + early stop
    if val_acc > best_val_acc + 1e-4:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_boxing_lstm.pth")
        wait = 0
        print(f"✔️  New best model saved (val_acc={val_acc:.3f})")
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f" Early stopping (no improvement for {PATIENCE} epochs)")
            break
