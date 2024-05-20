import numpy as np
from keras.callbacks import ModelCheckpoint

class SaveBestLoss(ModelCheckpoint):
    def __init__(self, filepath, txt_filepath, monitor='loss', save_best_only=True, verbose=0, mode='auto'):
        super(SaveBestLoss, self).__init__(filepath, monitor=monitor, save_best_only=save_best_only, verbose=verbose, mode=mode)
        self.txt_filepath = txt_filepath
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if self.save_best_only:
            if self.monitor_op(current, self.best):
                self.best = current
                self._save_best_loss_to_file()
        super(SaveBestLoss, self).on_epoch_end(epoch, logs)

    def _save_best_loss_to_file(self):
        with open(self.txt_filepath, 'w') as f:
            f.write(f"Best {self.monitor}: {self.best}\n")
