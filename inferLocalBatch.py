os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
import cv2
import tensorflow as tf
import numpy as np
import os
import glob
import pathlib
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import time

IMG_EXTS = ['*.jpg', '*.jpeg', '*.png']
IMAGE_PATHS =[]
[IMAGE_PATHS.extend(glob.glob(f'sample/*'+ x, recursive=True)) for x in IMG_EXTS]
image_file = "sample/hello_1.jpg"


MODELPATH = '/media/ezahan/DATA/workspace/tensorflow/object_detection/pretrained_models'
MODELNAME = 'faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8'

threshold =0.5

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# print("Version of Tensorflow: ", tf.__version__)
# print("Cuda Availability: ", tf.test.is_built_with_cuda())
# print("GPU Availability: ", tf.test.is_gpu_available())

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


################# DOWNLOAD AND EXTRACT MODEL
def download_model(model_name, model_date):
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name,
                                        origin=base_url + model_date + '/' + model_file,
                                        untar=True)
    return str(model_dir)

MODELDATE  = '20200711'
# PATH_TO_MODEL_DIR = download_model(MODELNAME, MODELDATE)

################# DOWNLOAD LABEL FILE
def download_labels(filename):
    base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
    label_dir = tf.keras.utils.get_file(fname=filename,
                                        origin=base_url + filename,
                                        untar=False)
    label_dir = pathlib.Path(label_dir)
    return str(label_dir)

LABELNAME = 'mscoco_complete_label_map.pbtxt'
classFile = "coco.names"
PATH_TO_LABEL = download_labels(LABELNAME)

################ LOAD THE MODEL
model_dir = MODELPATH + '/' + MODELNAME + '/saved_model'
print(model_dir)

# print("Loading Model: " + MODELNAME, end='')
start_time = time.time()

# tf.keras.backend.clear_session()
model = tf.saved_model.load(model_dir)

end_time = time.time()
elapsed_time = end_time - start_time

print("Model " + MODELNAME + " loaded successfully !!")
print("Time taken to load : {} seconds" .format(elapsed_time))

################ LOAD LABELMAP DATA (.pbtxt)

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABEL,
                                                                    use_display_name=True)


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
    path: the file path to the image

    Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
    """
    # return np.array(Image.open(path))
    return np.array(cv2.imread(path))


for image_path in IMAGE_PATHS:
    currentDir, imageFile = os.path.split(image_path)
    # print('Running inference for {}... \n'.format(image_path), end='')
   

    image_np = load_image_into_numpy_array(image_path)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    input_tensor = np.expand_dims(image_np, 0)
    detections = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=50,
          min_score_thresh=.50,
          agnostic_mode=False)
    
    print('Running inference for {}... \n'.format(imageFile), end='') 
    
    cv2.imwrite("output_batch/" + imageFile + "_predict.jpg", image_np_with_detections)
    # cv2.imshow("Result", image_np_with_detections)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows

