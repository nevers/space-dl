# space-dl
A convolutional neural network inspired by the YOLO algorithm to support spacecraft docking in space.

The working principles and results are detailed and discussed in this blog post: [TODO]

## Quickstart

### Obtain the YOLOv3 Keras model
1. Download YOLOv3 model description and weights from the [YOLO website](http://pjreddie.com/darknet/yolo/).
2. Convert the Darknet YOLO model description to a Keras model using [this script](https://github.com/qqwweee/keras-yolo3/blob/master/convert.py).
```
$ wget https://pjreddie.com/media/files/yolov3.weights
$ wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
$ python convert.py yolov3.cfg yolov3.weights .model/yolov3.h5
```

### Train and evaluate the model
1. Install python pip dependencies (consider using virtualenv for that).
```
$ virtualenv .venv
$ . .venv/bin/activate
$ pip install -r requirements
```
2. Simply run yolo3.py to start training and evaluation. It will automatically perform the following operations:
* download the dataset to ~/.datasets
* load the the pretrained model from yolov3.h5
* retrain the second half of the model whilst keeping the first half of the model parameters constant on your GPU.
* evaluate the model 2 times per epoch on your GPU
* write training and evalation metrics for tensorboard to .model/{train,eval} 
* write positive and negative evaluation images, annotated with their predictions to .model/eval-img
* periodically save the model state to .model/model

```
$ ./yolo3.py
```

2. Run tensor board to monitor the training and evaluation process.
```
$ tensorboard --host=0.0.0.0 --logdir=.model
```
Point your browser to [http://localhost:6006](http://localhost:6006)
