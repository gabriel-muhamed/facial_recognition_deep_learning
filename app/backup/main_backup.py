import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import pytz
import time

from keras.utils import Progbar
from keras.metrics import Recall, Precision
from datetime import datetime
import matplotlib.pyplot as plt

from app.utils.capture_image_data import CaptureImageData
from app.services.siamese_model import SiameseModel
from app.services.embedding import Embedding
from app.services.layer_distance import L1Distance

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

sp_tz = pytz.timezone('America/Sao_Paulo')

POS_PATH = os.path.join('data', 'positive')
ANC_PATH = os.path.join('data', 'anchor')
NEG_PATH = os.path.join('data', 'negative')

cid = CaptureImageData(POS_PATH, ANC_PATH)
embedding = Embedding()
l1_distance = L1Distance()
siamese_model = SiameseModel(embedding, l1_distance).__call__()

def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (105,105))
    img = img / 255.0
    return img

def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)

# Pega as imagens do diretório (Take the images from directory)
anchor = tf.data.Dataset.list_files(ANC_PATH+'\*.jpg').take(300)
positive = tf.data.Dataset.list_files(POS_PATH+'\*.jpg').take(300)
negative = tf.data.Dataset.list_files(NEG_PATH+'\*.jpg').take(300)

# Cria o grupo de imagens positivas (1) e negativas (0) e junta em um dataset
positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

# Para cada dupla de imagem, vai fazer o preprocessamento para o tamanho padrão.
data = data.map(preprocess_twin, num_parallel_calls=tf.data.AUTOTUNE)
# Aleatoriza os dados
data = data.cache()
data = data.shuffle(buffer_size=1024)

# Separa os dados de treinamento
train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

# Separa os dados de teste
test_data = data.skip(round(len(data)*.7))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4)

# siamese_model.summary()

checkpoint_dir = os.path.join('training_checkpoints', 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:
        x = batch[:2]
        y = batch[2] 

        yhat = siamese_model(x, training=True)

        loss = binary_cross_loss(y, yhat)

    grad = tape.gradient(loss, siamese_model.trainable_variables)

    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
    
    return loss

def train(train_data, epochs):
    now_sp = datetime.now(sp_tz)
    
    print(f'Começando treinamento da rede. Hora: {now_sp.time()}')
    
    for epoch in range(1, epochs+1):
        print('\n Epoch {}/{}'.format(epoch, epochs))
        progbar = Progbar(len(train_data))

        for idx, batch in enumerate(train_data):
            train_step(batch)
            progbar.update(idx+1)

        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_dir)

    print(f'Finalizando treinamento da rede. Hora: {now_sp.time()}')

status = checkpoint.restore(tf.train.latest_checkpoint('training_checkpoints')).expect_partial()

test_input, test_val, y_true = test_data.as_numpy_iterator().next()

y_hat = siamese_model.predict([test_input, test_val])
predictions = (['Semelhante' if prediction > 0.5 else 'Diferente' for prediction in y_hat])

for i in range(len(test_input)):
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(test_input[i])
    plt.subplot(1, 2, 2)
    plt.imshow(test_val[i])
    plt.title(f'Original: {int(y_true[i])} | Rede: {predictions[i]}')
    plt.show()
    time.sleep(1)
    plt.close()