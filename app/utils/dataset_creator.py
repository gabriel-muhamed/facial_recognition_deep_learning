import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import glob

from functools import reduce

class DatasetCreator():
    def create_dataset_from_classes(self, class_paths, samples_per_class=20):
        print('Creating datasets from models...')
        datasets = {}
        for name, path in class_paths.items():
            jpg_files = glob.glob(f"{path}/*.jpg")
            jpeg_files = glob.glob(f"{path}/*.jpeg")
            
            ds_list = []
            
            if jpg_files:
                ds_jpg = tf.data.Dataset.list_files(f"{path}/*.jpg", shuffle=True).take(samples_per_class)
                ds_list.append(ds_jpg)
                print(f"✅ {name}: {len(jpg_files)} arquivos .jpg encontrados")

            if jpeg_files:
                ds_jpeg = tf.data.Dataset.list_files(f"{path}/*.jpeg", shuffle=True).take(samples_per_class)
                ds_list.append(ds_jpeg)
                print(f"✅ {name}: {len(jpeg_files)} arquivos .jpeg encontrados")
                
            if len(ds_list) > 1:
                ds = ds_list[0].concatenate(ds_list[1])
            else:
                ds = ds_list[0]
            
            datasets[name] = ds
        print('Dataset created...')
        return datasets

    def make_pairs(self, datasets):
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