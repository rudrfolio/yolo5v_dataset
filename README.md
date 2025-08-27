1.use resizefinal.py and mention the dataset path(in SOURCE_DIR variable) and mention the desired size in mb (in TARGET_SIZE_MB variable),it will create a new folder "dataset_limited" and return the new resized dataset
2.now use autolabelfinal.py and give dataset_limited directory in (DATASET_DIR variable), now u will get dataset_final correctly labled and structured in yolov5 format and train/test split
3.now it can be used to train our yolov5 model
  steps to train yolo5v model:
  1.clone the yolov5 repository from github
  2.paste our dataset_final in same folder
  3.make a customD.yaml file in format of (coco128.yaml (default of yolov5))
  (a customD.yaml file is being uploaded in repository for reference)
  now paste it in yolo5v/data "here" 
  4.use the given train.py to train the model(note: give the path of yaml file correctly in it) adjust the epochs according to your choice.
  5.the trained model and all result will be saved in yolo5v/runs/train/exp...(specify latest exp no.here )
  6.now use detect.py give the source path of the new image on which u want to test correctly, u will get result here yolo5v/runs/detect/exp....(latest exp no.here )
