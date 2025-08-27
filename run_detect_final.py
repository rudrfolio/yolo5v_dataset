import subprocess
import cv2
from pathlib import Path
import shutil

# -----------------------------
# Settings
# -----------------------------
YOLO_DETECT = "yolov5/detect.py"
WEIGHTS = "yolov5/runs/train/exp11/weights/best.pt"
SOURCE = "image.png"
SAVE_DIR = "runs/detect/exp4"  
IMG_SIZE = "640"
CONF = "0.25"
# -----------------------------

# Clear output folder if it exists
save_path = Path(SAVE_DIR)
if save_path.exists() and save_path.is_dir():
    shutil.rmtree(save_path)
    print(f"Cleared previous outputs in {SAVE_DIR}")

# Run YOLOv5 detection in the same experiment folder
command = [
    "python", YOLO_DETECT,
    "--weights", WEIGHTS,
    "--source", SOURCE,
    "--img", IMG_SIZE,
    "--conf", CONF,
    "--project", "runs/detect",
    "--name", "exp4",
    "--exist-ok",  # overwrite same folder instead of creating new exp
    "--save-txt",
    "--save-conf"
]

subprocess.run(command, check=True)


cv2.imshow("abc",cv2.imread(f"{SAVE_DIR}/{SOURCE}"))
cv2.waitKey(0)
cv2.destroyAllWindows()
