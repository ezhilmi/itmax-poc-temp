
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import os
import urllib.request
import numpy as np
import cv2
import time
import threading, queue
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #Suppress Tensorflow Logging

tf.get_logger().setLevel('ERROR')       #Suppress Tensorflow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

MODELPATH = '/media/ezahan/DATA/workspace/tensorflow/object_detection/pretrained_models'
MODELNAME = 'centernet_resnet50_v2_512x512_kpts_coco17_tpu-8'
PATH_TO_CKPT = os.path.join(MODELPATH, os.path.join(MODELNAME, 'checkpoint/'))
PATH_TO_CFG = os.path.join(MODELPATH, os.path.join(MODELNAME, 'pipeline.config'))

CCTV_RTSP = 'rtsp://admin:senatraffic1234*@192.168.1.65/channels/101'
VIDEOPATH = '/media/ezahan/DATA/workspace/tensorflow/object_detection/vid_sample/run_jog.mp4'


# INPUT_QUEUE = queue.Queue()
# OUTPUT_QUEUE = queue.Queue()

threshold = 0.5

######### DOWNLOAD LABELS FILE
LABELNAME = 'mscoco_label_map.pbtxt'
LABELS_DOWNLOAD_BASE = \
    'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
PATH_TO_LABELS = os.path.join(MODELPATH, os.path.join(MODELNAME, LABELNAME))
if not os.path.exists(PATH_TO_LABELS):
    print('Downlading label file ...\n', end = '')
    urllib.request.urlretrieve(LABELS_DOWNLOAD_BASE + LABELNAME, PATH_TO_LABELS)
    print('Download label file is DONE')


######## LOAD THE MODEL

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load pipeline config and build a detection model

# start_time = time.time()

configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

# end_time = time.time()
# elapsed_time = end_time - start_time

print("Model " + MODELNAME + " loaded successfully !!")
# print("Time taken to load : {} seconds" .format(elapsed_time))

@tf.function
def detect_fn(image):
    """Detect objects in image"""

    image, shapes = detection_model.preprocess(image)
    prediction_dicts = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dicts, shapes)

    # OUTPUT_QUEUE.put(detections)
    return detections, prediction_dicts, tf.reshape(shapes, [-1])

######## LOAD LABEL MAP
# label_map_path = configs['eval_input_config'].label_map_path
# label_map = label_map_util.load_labelmap(label_map_path)
# categories = label_map_util.convert_label_map_to_categories(
#     label_map,
#     max_num_classes=label_map_util.get_max_label_map_index(label_map),
#     use_display_name=True
# )
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
# label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)
# print(category_index)

COCO17_HUMAN_POSE_KEYPOINTS = [(0, 1),
(0, 2),
(1, 3),
(2, 4),
(0, 5),
(0, 6),
(5, 7),
(7, 9),
(6, 8),
(8, 10),
(5, 6),
(5, 11),
(6, 12),
(11, 12),
(11, 13),
(13, 15),
(12, 14),
(14, 16)]

def rescale_frame(frame, scale):        # Works for image, camera, live video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimension = (width, height)
    return cv2.resize(frame, dimension, interpolation= cv2.INTER_AREA)

def get_keypoint_tuples(eval_config):
  """Return a tuple list of keypoint edges from the eval config.
  
  Args:
    eval_config: an eval config containing the keypoint edges
  
  Returns:
    a list of edge tuples, each in the format (start, end)
  """
  tuple_list = []
  kp_list = eval_config.keypoint_edge
  for edge in kp_list:
    tuple_list.append((edge.start, edge.end))
  return tuple_list


# ######### Start worker threads
# num_worker_threads = 4
# for _ in range(num_worker_threads):
#     t = threading.Thread(target=detect_fn)
#     t.daemon = True
#     t.start()

cap = cv2.VideoCapture(VIDEOPATH)
# t = threading.Thread(target=)
# t.start()

startTime = 0

while True:
    #Read frame from camera
    ret, frame = cap.read()
    image_np = np.array(frame)


    currentTime = time.time()
    fps = 1/(currentTime - startTime)
    startTime = currentTime

    #Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis = 0)

    #Things to try:
    #Flip horizontal
    #image_np = np.fliplr(image_np).copy()

    #Convert image to grayscale
    #image_np = np.title(
    #       np.mean(imaeg_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))

    detections = {key:value[0, : num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # Use keypoints if available in detections
    # keypoints, keypoint_scores = None, None
    # if 'detection_keypoints' in predictions_dict:
    #     keypoints = predictions_dict['detection_keypoints']
    #     keypoint_scores = predictions_dict['detection_keypoint_scores']

    np.set_printoptions(threshold=sys.maxsize)
    # print(detections.keys())
    # print(predictions_dict['detection_keypoints'])
    # different object detection models have additional results
    # all of them are explained in the documentation

    # result = {key:value.numpy() for key,value in detections.items()}
    # print(result.keys())

    label_id_offset = 1
    image_np_with_detections = image_np.copy()


    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        (detections['detection_classes']+label_id_offset).astype(int),
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=50,
        min_score_thresh=threshold,
        agnostic_mode=False,
        keypoints=predictions_dict,
        keypoint_scores=keypoint_scores,
        keypoint_edges=get_keypoint_tuples(configs['eval_config'])
        )

    #Resize Output
    frame_resized = rescale_frame(image_np_with_detections, scale = 1)

    #Show FPS
    cv2.putText(frame_resized, "FPS: " + str(int(fps)), (20,100), cv2.FONT_HERSHEY_SIMPLEX, 1 , (0,255,0), 3)

    #Display output
    # cv2.imshow('object detection', image_np)
    cv2.imshow('object detection', frame_resized)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        break

cv2.destroyAllWindows