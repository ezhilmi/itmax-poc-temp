import cv2
import numpy as np
import tensorflow as tf
import os
from object_detection.utils import visualization_utils as vis_util
import urllib.request

# Path to the frozen inference graph and label map
# PATH_TO_FROZEN_GRAPH = 'path/to/your/frozen_inference_graph.pb'
# PATH_TO_LABELS = 'path/to/your/label_map.pbtxt'
MODELPATH = '/media/ezahan/DATA/workspace/tensorflow/object_detection/pretrained_models'
MODELNAME = 'centernet_resnet50_v2_512x512_kpts_coco17_tpu-8'
PATH_TO_MODEL = os.path.join(MODELPATH, os.path.join(MODELNAME, 'saved_model/saved_model.pb'))
PATH_TO_CKPT = os.path.join(MODELPATH, os.path.join(MODELNAME, 'checkpoint/'))
PATH_TO_CFG = os.path.join(MODELPATH, os.path.join(MODELNAME, 'pipeline.config'))

######### DOWNLOAD LABELS FILE
LABELNAME = 'mscoco_label_map.pbtxt'
LABELS_DOWNLOAD_BASE = \
    'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
PATH_TO_LABELS = os.path.join(MODELPATH, os.path.join(MODELNAME, LABELNAME))
if not os.path.exists(PATH_TO_LABELS):
    print('Downlading label file ...\n', end = '')
    urllib.request.urlretrieve(LABELS_DOWNLOAD_BASE + LABELNAME, PATH_TO_LABELS)
    print('Download label file is DONE')


# Load the frozen graph and label map
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Load the label map
label_map = {}
with open(PATH_TO_LABELS, 'r') as f:
    for line in f.readlines():
        if 'id' in line:
            id_ = int(line.strip().split(':')[-1])
        elif 'display_name' in line:
            display_name = line.strip().split(':')[-1].strip().strip('"')
            label_map[id_] = display_name

# Keypoint detection function
def detect_keypoints(image, sess, detection_graph):
    image_np_expanded = np.expand_dims(image, axis=0)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    detection_keypoints = detection_graph.get_tensor_by_name('detection_keypoints:0')

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        label_map,
        use_normalized_coordinates=True,
        line_thickness=2,
        min_score_thresh=0.5)

    # Get keypoints and draw them on the image
    keypoints = np.squeeze(sess.run(detection_keypoints, feed_dict={image_tensor: image_np_expanded}))
    vis_util.draw_keypoints_on_image_array(
        image,
        keypoints,
        min_score_thresh=0.5,
        use_normalized_coordinates=True,
        line_thickness=2)

    return image

# Load the video
VIDEOPATH = '/media/ezahan/DATA/workspace/tensorflow/object_detection/vid_sample/nyc_walk.mp4'
cap = cv2.VideoCapture(VIDEOPATH)

# Create a session and start inference
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform inference and display the output frame
            output_frame = detect_keypoints(frame, sess, detection_graph)
            cv2.imshow('Object Detection', output_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
