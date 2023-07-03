from DetectorURL import *

modelURL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8.tar.gz'

image_path = "sample/hello_1.jpg"
classFile = "coco.names"
threshold =0.3

detector = DetectorURL()
detector.readClasses(classFile)
detector.downloadModel(modelURL)
detector.loadModelURL()
detector.predictImage(image_path, threshold)