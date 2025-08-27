# YOLOv5 Pest Detection – Complete Training & Inference Guide

This repository provides a complete pipeline to prepare, label, train, and test a YOLOv5 model on a custom pest dataset.

---

## 📌 Steps to Prepare and Train

### 1. Resize Dataset

Use `resizefinal.py` to resize your dataset.

* Set `SOURCE_DIR` = path to your dataset
* Set `TARGET_SIZE_MB` = desired dataset size (in MB)

```bash
python resizefinal.py
```

✅ Output: A new folder `dataset_limited`

### 2. Auto-Label Dataset

Use `autolabelfinal.py` to structure your dataset in YOLOv5 format.

* Set `DATASET_DIR = "dataset_limited"`

```bash
python autolabelfinal.py
```

✅ Output: A new folder `dataset_final` (YOLOv5 format + train/val split)

### 3. Clone YOLOv5 Repository

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
```

### 4. Place Dataset

Move `dataset_final` inside the YOLOv5 root folder:

```
yolov5/
 ├── dataset_final/
 ├── data/
 ├── models/
 ├── train.py
 ├── detect.py
```

### 5. Create Custom YAML

Create a file `customD.yaml` (similar to `coco128.yaml`) and configure:

```yaml
train: ./dataset_final/images/train
val: ./dataset_final/images/val

nc: <num_classes>
names: [ 'class1', 'class2', ... ]
```

Save it inside `yolov5/data/customD.yaml`.

### 6. Train Model

Run training with your dataset:

```bash
python train.py --img 640 --batch 16 --epochs 50 --data data/customD.yaml --weights yolov5s.pt --cache
```

✅ Output: Training results + weights saved in `yolov5/runs/train/expX/`

### 7. Run Inference

Run detection on new images:

```bash
python detect.py --weights runs/train/expX/weights/best.pt --img 640 --conf 0.25 --source path/to/image_or_folder
```

✅ Output: Results saved in `yolov5/runs/detect/expY/`

---

## 📊 Workflow Diagram (Mermaid)

```mermaid
flowchart TD
    A[Raw Dataset] -->|Step 1: resizefinal.py| B[dataset_limited]
    B -->|Step 2: autolabelfinal.py| C[dataset_final in YOLOv5 format]
    C -->|Step 3: Clone Repo| D[YOLOv5 Repo]
    D -->|Step 4: Place dataset_final| E[YOLOv5 Folder]
    E -->|Step 5: Create YAML| F[customD.yaml in data/]
    F -->|Step 6: Train| G[Trained Weights in runs/train/expX]
    G -->|Step 7: Detect| H[Results in runs/detect/expY]
```

---

## ✅ Notes

* Adjust `--epochs` in training as per your dataset size.
* Ensure `customD.yaml` paths match your folder structure.
* Use the latest trained `expX` folder for detection.
* Confidence threshold in `detect.py` can be adjusted with `--conf`.
