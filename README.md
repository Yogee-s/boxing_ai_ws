# Boxing AI Pipeline

This project builds a pose-based boxing-move classifier end-to-end: from raw video to live webcam inference. Below is an overview of each stage, directory layout, and how to reproduce or extend the pipeline.

---

## 🚀 Overview

1. **Data acquisition**
   Collect raw boxing videos (sparring, bag work, punch drills).

2. **Timestamp labeling & segmentation**
   Manually label start/end times of individual moves in an Excel sheet, then automatically slice your raw videos into fixed-length clips (MP4s) with `data_processing.ipynb` / `training.py`.

3. **Pose extraction**
   Run a YOLOv11-pose model on each clip to extract per-frame 17 keypoints → save as NumPy arrays (`.npy`) and generate a `labels.csv`.

4. **Visualization (optional)**
   Overlay keypoints and skeleton on videos to sanity-check your pose data.

5. **Train temporal classifier**
   Use sliding windows of pose sequences to train an LSTM that maps a sequence of 50 frames → one of eight move classes.

6. **Live inference**
   Load the trained LSTM + YOLO-pose and predict moves in real time from your webcam.

---

## 📂 Directory Structure

```
BOXING_AI_WS/
├── data_raw/                      # Your uncut source videos
│   ├── punching_bag_fast/
│   ├── repeated_punches/
│   └── sparring_pov/
│
├── time_stamps/
│   └── Boxing_Videos_Timestamp.xlsx  # Excel with manually labeled Start/End times
│
├── dataset/                       # Auto-cut MP4 clips, organized by label
│   ├── block/
│   ├── cross/
│   ├── idle/
│   ├── jab/
│   ├── left_hook/
│   ├── left_uppercut/
│   ├── right_hook/
│   └── right_uppercut/
│
├── dataset_with_poses/            # NumPy pose arrays + labels.csv
│   ├── block/          *.npy      # one .npy per clip, shape (T,34)
│   ├── ...
│   └── labels.csv      mapping from npy_path → label
│
├── dataset_with_poses_visualization/
│   └── block/
│       ├── *_annotated.mp4        # sample pose-overlaid videos (sanity check)
│
├── models/                        # (optional) store custom weights or metadata
│
├── best_boxing_lstm.pth           # Trained LSTM checkpoint (best on validation)
│
├── data_processing.ipynb          # Jupyter notebook for cutting clips & extracting poses
├── training.ipynb                 # Notebook for training LSTM interactively
│
├── training.py                    # Standalone training script (train_boxing_moves.py)
├── run_inference.py               # Real-time webcam demo script
│
└── README.md                      # ← you are here
```

---

## 🛠️ Prerequisites

- **Python 3.8+**
- **FFmpeg** (for video cutting & imageio)
- **pip packages**

  ```bash
  pip install ultralytics torch torchvision \
              opencv-python-headless imageio \
              numpy pandas scikit-learn
  ```

  > **Note:** If you want GUI windows (`cv2.imshow`) in `run_inference.py`, install Ubuntu’s `python3-opencv` instead of the headless wheel.

---

## ▶️ Step-by-Step Usage

### 1. Segment your raw videos

1. Open `time_stamps/Boxing_Videos_Timestamp.xlsx`
2. Label start/end times for each move in one video (for now).
3. Run:

   ```bash
   python training.py
   ```

   or execute cells in `data_processing.ipynb`.

4. Check `dataset/<label>/*.mp4` for your cut clips.

### 2. Extract poses

```bash
# In a Jupyter cell or script:
# configure paths at top of data_processing.ipynb:
#   POSE_MODEL_PATH, DATASET_ROOT, OUTPUT_ROOT
# then run the cell to populate dataset_with_poses/
```

- VM: `dataset_with_poses_visualization/` will contain a few annotated MP4s.

### 3. Train your LSTM model

```bash
python training.py
```

Or in Jupyter:

1. Open `training.ipynb`.
2. Run cells 1–4 to prepare data, define model, and loaders.
3. Run the **Training** cell (with sliding windows, augmentation, early stopping).
4. Your best checkpoint will be saved as `best_boxing_lstm.pth`.

### 4. Real-time inference demo

```bash
python run_inference.py
```

- Opens your laptop camera, overlaying skeleton+predicted move.
- Press **q** to quit.

**Tip:** To test on a saved video, replace the `VideoCapture(0)` line with your MP4 path.

---

## 🎯 Next Steps

- **Label more data** in `time_stamps` → bigger `dataset/`
- **Tune hyperparameters** (window length, step, hidden size, dropout)
- **Try alternative models** (Temporal CNN, Transformer)
- **Deploy** the ONNX export of your LSTM for embedded devices

---
