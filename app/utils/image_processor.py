import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

class ImageProcessor:
    def __init__(self):
        self._data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ])
    
    def preprocess(self, file_path):
        byte_img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(byte_img, channels=3)
        img = tf.image.resize(img, (105,105))
        img = img / 255.0
        return img

    def preprocess_twin(self, input_img, validation_img, label):
        return(self.preprocess(input_img), self.preprocess(validation_img), label)