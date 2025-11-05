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


ROCK_PATH = os.path.join('data', 'chris_rock')
BALE_PATH = os.path.join('data', 'christian_bale')
DUSTIN_PATH = os.path.join('data', 'dustin_hoffman')
GARY_PATH = os.path.join('data', 'gary_oldman')
HEATH_PATH = os.path.join('data', 'heath_ledger')
HUGH_PATH = os.path.join('data', 'hugh_jackman')
JACKIE_PATH = os.path.join('data', 'jackie_chan')
SAIMON_PATH = os.path.join('data', 'saimon')

class_paths = {
    "rock": ROCK_PATH,
    "bale": BALE_PATH,
    "dustin": DUSTIN_PATH,
    "gary": GARY_PATH,
    "heath": HEATH_PATH,
    "hugh": HUGH_PATH,
    "jackie": JACKIE_PATH,
    "saimom": SAIMON_PATH
}

datasets = dtc.create_dataset_from_classes(class_paths, 80)
data = dtc.make_pairs(datasets)

data = data.map(imgp.preprocess_twin, num_parallel_calls=tf.data.AUTOTUNE)
data = data.cache()
data = data.shuffle(buffer_size=1024)

train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16).prefetch(8)

test_data = data.skip(round(len(data)*.7))
test_data = test_data.batch(16).prefetch(8)

binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = os.path.join('checkpoints', 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

trainer.train(
    train_data,
    50,
    checkpoint,
    checkpoint_dir,
    siamese_model,
    binary_cross_loss,
    opt
)

# Restore last checkpoint
# get_checkpoint_dir = os.path.join('checkpoints')
# latest = tf.train.latest_checkpoint(get_checkpoint_dir)
# if latest:
#     checkpoint.restore(latest).expect_partial()
#     print(f"✅ Checkpoint restaurado: {latest}")
# else:
#     print("⚠️ Nenhum checkpoint encontrado.")

# test_input, test_val, y_true = next(test_data.as_numpy_iterator())

# pred = siamese_model.predict([test_input, test_val])

# for i in range(len(test_input)):
#     plt.figure(figsize=(8, 4))

#     plt.subplot(1, 2, 1)
#     plt.imshow(test_input[i])
#     plt.title("Imagem 1")
#     plt.axis("off")

#     plt.subplot(1, 2, 2)
#     plt.imshow(test_val[i])
#     plt.title(f"Imagem 2\nLabel: {y_true[i]}\nPred: {pred[i]}")
#     plt.axis("off")

#     plt.tight_layout()
#     plt.show()
