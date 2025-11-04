import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pytz
import tensorflow as tf

from functools import reduce
from keras.callbacks import EarlyStopping, ModelCheckpoint

from layer_distance import L1Distance

l1_distance = L1Distance()

siamese_model = tf.keras.models.load_model('model/siamesemodel.h5', custom_objects={'L1Distance': L1Distance})

sp_tz = pytz.timezone('America/Sao_Paulo')

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img, channels=3)
    img = tf.image.resize(img, (105,105))
    img = data_augmentation(img)
    img = img / 255.0
    return img

def preprocess_twin(input_img, validation_img, label):
    return((preprocess(input_img), preprocess(validation_img)), label)

def create_dataset_from_classes(class_paths, samples_per_class=20):
    print('Creating datasets from models...')
    datasets = {}
    for name, path in class_paths.items():
        ds = tf.data.Dataset.list_files(os.path.join(path, '*jpg')).take(samples_per_class)
        datasets[name] = ds
    print('Dataset created...')
    return datasets

def make_pairs(datasets):
    print('Creating paired datasets...')
    
    pairs = []
    
    # Creating positive pairs (between equal classes)
    for _, ds in datasets.items():
        ds_positive = tf.data.Dataset.zip((
            ds,
            ds.shuffle(100, reshuffle_each_iteration=True),
            tf.data.Dataset.from_tensor_slices(tf.ones(len(ds)))
        ))
        pairs.append(ds_positive)
    print('Positive pairs created...')

    # Creating negative pairs (between differents classes)
    class_names = list(datasets.keys())
    for i in range(len(class_names)):
        for j in range(i + 1, len(class_names)):
            ds1 = datasets[class_names[i]]
            ds2 = datasets[class_names[j]]
            ds_negative = tf.data.Dataset.zip((
                ds1,
                ds2,
                tf.data.Dataset.from_tensor_slices(tf.zeros(len(ds1)))
            ))
            pairs.append(ds_negative)
    print('Negative pairs created...')
    
    # Concatenated datasets
    return reduce(lambda x, y: x.concatenate(y), pairs)

# Caminhos
NEY_PATH = os.path.join('application_data', 'data', 'neymar')
MESSI_PATH = os.path.join('application_data', 'data', 'messi')
CR7_PATH = os.path.join('application_data', 'data', 'cr7')
VINIJR_PATH = os.path.join('application_data', 'data', 'vinijr')

class_paths = {
    "neymar": NEY_PATH,
    "messi": MESSI_PATH,
    "cr7": CR7_PATH,
    "vinijr": VINIJR_PATH
}

datasets = create_dataset_from_classes(class_paths)
data = make_pairs(datasets)

# Para cada dupla de imagem, vai fazer o preprocessamento para o tamanho padrão.
data = data.map(preprocess_twin, num_parallel_calls=tf.data.AUTOTUNE)
# Aleatoriza os dados
data = data.cache()
data = data.shuffle(buffer_size=1024)

# Separa os dados de treinamento
train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16).prefetch(8)

# Separa os dados de teste
test_data = data.skip(round(len(data)*.7))
test_data = test_data.batch(16).prefetch(8)

binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-3)

# Verifique
siamese_model.summary()
siamese_model.compile(
    optimizer=opt,
    loss=binary_cross_loss,
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

cb_early = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
cb_ckpt = ModelCheckpoint('model/siamesemodel_finetuned_v6.h5', monitor='val_loss', save_best_only=True)

siamese_model.fit(train_data, validation_data=test_data, epochs=20, callbacks=[cb_early, cb_ckpt])

