import subprocess

# Command to run YOLOv5 training
command = [
    "python", "yolov5/train.py",
    "--img", "640",
    "--batch", "16",
    "--epochs", "3",
    "--data", "yolov5/data/customD.yaml",
    "--weights", "yolov5s.pt",
    "--cache",
    "--workers", "0"
]

# Execute the command
subprocess.run(command)
