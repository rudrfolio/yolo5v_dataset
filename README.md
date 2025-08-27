1.use resizefinal.py and mention the dataset path(in SOURCE_DIR variable) and mention the desired size in mb (in TARGET_SIZE_MB variable),it will create a new folder "dataset_limited" and return the new resized dataset
2.now use autolabelfinal.py and give dataset_limited directory in (DATASET_DIR variable), now u will get dataset_final correctly labled and structured in yolov5 format and train/test split
3.now it can be used to train our yolov5 model
  steps to train yolo5v model:
  
