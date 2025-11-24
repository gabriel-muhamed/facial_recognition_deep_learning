import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import pytz

from keras.utils import Progbar
from datetime import datetime


class Trainer:
    def __init__(self, patience=5):
        self._sp_tz = pytz.timezone('America/Sao_Paulo')
        
        self._metric_loss = tf.keras.metrics.Mean(name='loss')
        self._metric_acc = tf.keras.metrics.BinaryAccuracy(name='Accuracy')
        self._metric_prec = tf.keras.metrics.Precision(name='precision')
        self._metric_rec = tf.keras.metrics.Recall(name='recall')

        self.best_val_loss = float("inf")
        self.patience = patience
        self.wait = 0
        self.best_weights = None
    
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
    
    def _val_step(self, batch, model, loss_func):
        images = batch[:2]
        true_label = batch[2]

        predict_label = model(images, training=False)
        loss = loss_func(true_label, predict_label)
        return loss

    def train(self, train_data, val_data, epochs, checkpoint, checkpoint_dir, model, loss_func, opt):
        now_sp = datetime.now(self._sp_tz)
        print(f'🕔 Starting network training. Schedule: {now_sp.time()}')
        
        for epoch in range(1, epochs+1):
            print('\n Epoch {}/{}'.format(epoch, epochs))
            progbar = Progbar(len(train_data))

            self._metric_loss.reset_state()
            self._metric_acc.reset_state()
            self._metric_prec.reset_state()
            self._metric_rec.reset_state()

            # Training Loop
            for idx, batch in enumerate(train_data):
                self._train_step(batch, model, loss_func, opt)
                progbar.update(idx + 1, [
                    ('loss', self._metric_loss.result().numpy()),
                    ('acc', self._metric_acc.result().numpy()),
                    ('prec', self._metric_prec.result().numpy()),
                    ('rec', self._metric_rec.result().numpy())
                ])
            
            # Validation loop
            val_loss = 0.0
            for batch in val_data:
                val_loss += self._val_step(batch, model, loss_func)
            val_loss /= len(val_data)

            print(f" -> Epoch {epoch} metrics: "
                  f"loss={self._metric_loss.result():.4f}, "
                  f"acc={self._metric_acc.result():.4f}, "
                  f"prec={self._metric_prec.result():.4f}, "
                  f"rec={self._metric_rec.result():.4f}, "
                  f"val_loss={val_loss}")

            # Early Stropping
            if val_loss < self.best_val_loss:
                print("Val loss upgrade! Saving best model")
                self.best_val_loss = val_loss
                self.best_weights = model.get_weights()
                self.wait = 0
            else:
                self.wait += 1
                print(f"No upgrade ({self.wait}/{self.patience})")

                if self.wait >= self.patience:
                    print("Early Stopping activated! Restoring best weights")
                    model.set_weights(self.best_weights)
                    checkpoint.save(file_prefix=checkpoint_dir + "/best")
                    break

            if epoch % 10 == 0:
                checkpoint.save(file_prefix=checkpoint_dir)

        now_sp = datetime.now(self._sp_tz)
        print(f'🕔 Finishing network training. Schedule: {now_sp.time()}')