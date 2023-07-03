import tensorflow as tf
import numpy as np
import pathlib

class LocalDetector:
    def __init__(self):
        pass

    def readClasses(self, classesFilePath):
        with open(classesFilePath, 'r') as f:
            self.classesList = f.read().splitlines()

            #Color List
            self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))



# Download labels file
    def download_labels(self, filename):
        base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
        label_dir = tf.keras.utils.get_file(fname=filename,
                                            origin=base_url + filename,
                                            untar=False)
        label_dir = pathlib.Path(label_dir)
        return str(label_dir)
    
