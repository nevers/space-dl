# space-dl
A convolutional neural network inspired by the YOLO algorithm to support spacecraft docking in space.


## Quickstart

### Obtain the YOLOv3 Keras model
1. Download YOLOv3 model description and weights from the [YOLO website](http://pjreddie.com/darknet/yolo/).
2. Convert the Darknet YOLO model description to a Keras model using [this script](https://github.com/qqwweee/keras-yolo3/blob/master/convert.py).
```
wget https://pjreddie.com/media/files/yolov3.weights
wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
```
