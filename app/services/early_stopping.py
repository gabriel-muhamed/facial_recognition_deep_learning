import numpy as np

class EarlyStoppingCallback:
    def __init__(self, patience = 10, wait = 0, best_weights = None, min_delta=1e-4, restore_best_weights = True):
        self.best_val_loss = float("inf")
        self.patience = patience
        self.wait = wait
        self.best_weights = best_weights
        self.restore_best_weights = restore_best_weights
        self.min_delta = min_delta
    
    def _compute_val_loss(self, val_data, model, loss_func):
        # Validation loop
        val_losses = []
        for batch in val_data:
            if len(batch) == 2:
                (img1, img2), y_true = batch
            else:
                img1, img2, y_true = batch
            y_pred = model([img1, img2], training=False)
            loss = loss_func(y_true, y_pred)
            val_losses.append(loss.numpy().mean())
        return np.mean(val_losses)
    
    def __call__(self, epoch, model, val_data, loss_func, checkpoint=None, checkpoint_dir=None):
        current_val_loss = self._compute_val_loss(val_data, model, loss_func)
        print(f"val_loss: {current_val_loss:.5f} ", end="")
        
        if epoch >= 10:
            # Early Stropping
            if self.current_val_loss < self.best_val_loss - self.min_delta:
                improvement = self.best_val_loss - current_val_loss
                print(f"Val loss upgrade! Saving best model ({self.best_val_loss:.5f} → {current_val_loss:.5f}, +{improvement:.5f})")
                
                self.best_val_loss = current_val_loss
                self.best_weights = model.get_weights()
                self.wait = 0
            else:
                self.wait += 1
                print(f"No upgrade ({self.wait}/{self.patience})")

                if self.wait >= self.patience:
                    print("Early Stopping activated! Restoring best weights")
                    print(f"Best val_loss: {self.best_val_loss:.5f}")

                    if self.restore_best_weights and self.best_weights is not None:
                        model.set_weights(self.best_weights)
                        checkpoint.save(file_prefix=checkpoint_dir + "/best")
                    return True
        return False