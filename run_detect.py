import subprocess

command = [
    "python", "yolov5/detect.py",
    "--weights", "yolov5/runs/train/exp4/weights/best.pt",
    "--img", "640",
    "--conf", "0.25",
    "--source", "a.jpg"  # folder or image path
]

# Run the command
subprocess.run(command)
# Get latest output image path

