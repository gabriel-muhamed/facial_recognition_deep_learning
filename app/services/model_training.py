import os
## Suppress TensorFlow logging (1 = filter INFO, 2 = filter WARNINGs too) ##
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import matplotlib.pyplot as plt

## Project-specific modules ##
from app.utils.dataset_creator import DatasetCreator
from app.utils.image_processor import ImageProcessor
from app.services.layer_distance import L1Distance
from app.services.embedding import Embedding
from app.services.siamese_model import SiameseModel
from app.services.trainer import Trainer

## GPU Configuration and retrieving the list of visible GPUs ##
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

## Model components instantiation ##
embedding = Embedding()
l1_distance = L1Distance()
dtc = DatasetCreator()
imgp = ImageProcessor()
trainer = Trainer(patience=25)
siamese_model = SiameseModel(embedding, l1_distance).__call__()

## Creating the class dataset ##
data_dir = 'data'
class_paths = {
    class_name: os.path.join(data_dir, class_name)
    for max_num_classes, class_name in enumerate(os.listdir(data_dir))
    if os.path.isdir(os.path.join(data_dir, class_name)) and max_num_classes < 30
}
datasets = dtc.create_dataset_from_classes(class_paths, 60)
data = dtc.make_pairs(datasets, 1750)
data = data.map(imgp.preprocess_twin, num_parallel_calls=tf.data.AUTOTUNE)
data = data.shuffle(buffer_size=1024)

## Separating training and test data ##
train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16).prefetch(tf.data.AUTOTUNE)
test_data = data.skip(round(len(data)*.7))
test_data = test_data.batch(16).prefetch(tf.data.AUTOTUNE)

binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4)

## Defining the checkpoint class and location ##
checkpoint_dir = os.path.join('checkpoints', 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

## Training ##
trainer.train(
    train_data,
    test_data,
    200,
    checkpoint,
    checkpoint_dir,
    siamese_model,
    binary_cross_loss,
    opt
)
