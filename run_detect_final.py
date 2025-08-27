import subprocess
import cv2
from pathlib import Path

# -----------------------------
# Settings
# -----------------------------
YOLO_DETECT = "yolov5/detect.py"
WEIGHTS = "yolov5/runs/train/exp11/weights/best.pt"
SOURCE = "a.jpg"
SAVE_DIR = "runs/detect/exp4"  # same as training exp
IMG_SIZE = "640"
CONF = "0.25"
# -----------------------------

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

# Display results
for img_path in Path(SAVE_DIR).glob("*.jpg"):
    img = cv2.imread(str(img_path))
    cv2.imshow("YOLO Detection", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
print(f"Detection complete! Results saved in {SAVE_DIR}")
