import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import matplotlib.pyplot as plt

from app.utils.dataset_creator import DatasetCreator
from app.utils.image_processor import ImageProcessor
from app.services.layer_distance import L1Distance
from app.services.embedding import Embedding
from app.services.siamese_model import SiameseModel
from app.services.trainer import Trainer

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

embedding = Embedding()
l1_distance = L1Distance()
dtc = DatasetCreator()
imgp = ImageProcessor()
trainer = Trainer()
siamese_model = SiameseModel(embedding, l1_distance).__call__()

data_dir = 'data'

class_paths = {
    class_name: os.path.join(data_dir, class_name)
    for max_num_classes, class_name in enumerate(os.listdir(data_dir))
    if os.path.isdir(os.path.join(data_dir, class_name)) and max_num_classes < 5
}

datasets = dtc.create_dataset_from_classes(class_paths, 60)
data = dtc.make_pairs(datasets, 100)

data = data.map(imgp.preprocess_twin, num_parallel_calls=tf.data.AUTOTUNE)
data = data.shuffle(buffer_size=1024)

test_data = data.skip(round(len(data)*.7))
test_data = test_data.batch(16).prefetch(tf.data.AUTOTUNE)

# Restore last checkpoint
opt = tf.keras.optimizers.Adam(1e-4)
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)
get_checkpoint_dir = os.path.join('checkpoints')
latest = tf.train.latest_checkpoint(get_checkpoint_dir)
if latest:
    checkpoint.restore(latest).expect_partial()
    print(f"✅ Checkpoint restored: {latest}")
else:
    print("⚠️ No checkpoints found.")

test_input, test_val, y_true = next(test_data.as_numpy_iterator())

pred = siamese_model.predict([test_input, test_val])

for i in range(len(test_input)):
    print(f'y_hat: {pred[i]}; y: {y_true[i]}')
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(tf.keras.utils.array_to_img(test_input[i]))
    plt.title("Image 1")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(tf.keras.utils.array_to_img(test_val[i]))
    plt.title(f"Image 2\nLabel: {y_true[i]}\nPred: {1 if pred[i] > 0.5 else 0}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()