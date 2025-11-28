class EarlyStopping:
    def __init__(patience = 10, wait = 0, best_weights):
        self.best_val_loss = float("inf")
        self.patience = patience
        self.wait = wait
        self.best_weights = best_weights
    
    def _val_step(self, batch, model, loss_func):
        images = batch[:2]
        true_label = batch[2]

        predict_label = model(images, training=False)
        loss = loss_func(true_label, predict_label)
        return loss
    
    def _get_val_loss(self, )
        # Validation loop
        val_metric = tf.keras.metrics.Mean()
        for batch in val_data:
            loss = self._val_step(batch, model, loss_func)
            val_metric.update_state(loss)
        val_loss = val_metric.result()