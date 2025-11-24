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
    for class_name in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, class_name))
}

datasets = dtc.create_dataset_from_classes(class_paths, 60)
data = dtc.make_pairs(datasets, 5800)

data = data.map(imgp.preprocess_twin, num_parallel_calls=tf.data.AUTOTUNE)
data = data.shuffle(buffer_size=1024)

train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(4).prefetch(2)

test_data = data.skip(round(len(data)*.7))
test_data = test_data.batch(4).prefetch(2)

binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = os.path.join('checkpoints-v4', 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

# trainer.train(
#     train_data,
#     test_data,
#     50,
#     checkpoint,
#     checkpoint_dir,
#     siamese_model,
#     binary_cross_loss,
#     opt
# )

# Restore last checkpoint
get_checkpoint_dir = os.path.join('checkpoints-v4')
latest = tf.train.latest_checkpoint(get_checkpoint_dir)
if latest:
    checkpoint.restore(latest).expect_partial()
    print(f"✅ Checkpoint restaurado: {latest}")
else:
    print("⚠️ Nenhum checkpoint encontrado.")

test_input, test_val, y_true = next(test_data.as_numpy_iterator())

# img1_path = os.path.join('data', 'test', 'm1.jpg')
# img2_path = os.path.join('data', 'test', 'n1.jpg')

# img1, img2 = imgp.preprocess(img1_path), imgp.preprocess(img2_path)

# img1 = tf.expand_dims(img1, axis=0)  # shape: (1, 105,105,3)
# img2 = tf.expand_dims(img2, axis=0)  # shape: (1, 105,105,3)

pred = siamese_model.predict([test_input, test_val])

for i in range(len(test_data)):
    print(f'y_hat: {pred[i]}; y: {y_true[i]}')
    # plt.figure(figsize=(8, 4))

    # plt.subplot(1, 2, 1)
    # plt.imshow(tf.keras.utils.array_to_img(img1[0]))
    # plt.title("Imagem 1")
    # plt.axis("off")

    # plt.subplot(1, 2, 2)
    # plt.imshow(tf.keras.utils.array_to_img(img2[0]))
    # plt.title(f"Imagem 2\nLabel: {0}\nPred: {pred}")
    # plt.axis("off")

    # plt.tight_layout()
    # plt.show()
