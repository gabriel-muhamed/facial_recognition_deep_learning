import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import pytz

from keras.utils import Progbar
from datetime import datetime


class Trainer:
    def __init__(self):
        self._sp_tz = pytz.timezone('America/Sao_Paulo')
        
        self._metric_loss = tf.keras.metrics.Mean(name='loss')
        self._metric_acc = tf.keras.metrics.BinaryAccuracy(name='Accuracy')
        self._metric_prec = tf.keras.metrics.Precision(name='precision')
        self._metric_rec = tf.keras.metrics.Recall(name='recall')
    
    @tf.function
    def _train_step(self, batch, model, loss_func, opt):
        images = batch[:2]
        true_label = batch[2] 
        
        with tf.GradientTape() as tape:
            predict_label = model(images, training=True)
            loss = loss_func(true_label, predict_label)

        grad = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grad, model.trainable_variables))
        
        self._metric_loss.update_state(loss)
        self._metric_acc.update_state(true_label, predict_label)
        self._metric_prec.update_state(true_label, predict_label)
        self._metric_rec.update_state(true_label, predict_label)
        
        return loss

    def train(self, train_data, epochs, checkpoint, checkpoint_dir, model, loss_func, opt):
        now_sp = datetime.now(self._sp_tz)
        print(f'Starting network training. Schedule: {now_sp.time()}')
        
        for epoch in range(1, epochs+1):
            print('\n Epoch {}/{}'.format(epoch, epochs))
            progbar = Progbar(len(train_data))

            self._metric_loss.reset_state()
            self._metric_acc.reset_state()
            self._metric_prec.reset_state()
            self._metric_rec.reset_state()

            for idx, batch in enumerate(train_data):
                self._train_step(batch, model, loss_func, opt)
                progbar.update(idx + 1, [
                    ('loss', self._metric_loss.result().numpy()),
                    ('acc', self._metric_acc.result().numpy()),
                    ('prec', self._metric_prec.result().numpy()),
                    ('rec', self._metric_rec.result().numpy())
                ])

            print(f" -> Epoch {epoch} metrics: "
                  f"loss={self._metric_loss.result():.4f}, "
                  f"acc={self._metric_acc.result():.4f}, "
                  f"prec={self._metric_prec.result():.4f}, "
                  f"rec={self._metric_rec.result():.4f}")

            if epoch % 10 == 0:
                checkpoint.save(file_prefix=checkpoint_dir)

        now_sp = datetime.now(self._sp_tz)
        print(f'Finishing network training. Schedule: {now_sp.time()}')