# AutoDrive
This is a supplementary auto driving demonstration for [Oceanbotech](http://www.oceanbotech.com/) SmartCar mobile platform.
## Requirements
- python 3.5+
- pytorch 1.0.0+
- PyQt
- torchvision
- opencv-python 3.4.0+
- numpy
- imutils
- matplotlib

## SignDetection

### Dataset
The dataset can be download from [BaiduCloud]()
Please extract SignDetectionDataset.zip and move **video** and **data** under the **SignDetection**

### Training
```shell script
cd SignDetection/scripts
python train.py
tensorboard --logdir="../logs" --port=7001
```

### Run
- Run with video
```shell script
cd SignDetection/scripts
python video_detect.py
```

- Run with Oceanbotech SmartCar
```shell script
cd AutoDriveGui
python gui.py
```
<div align=center><img src="https://github.com/Merical/AutoDrive/blob/master/Images/signdetection.png" width=640 height=480></div>
