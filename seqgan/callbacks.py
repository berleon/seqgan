import numpy as np
import keras


class EpochTextSampler(keras.callbacks.Callback):
    def __init__(self, data, X, batch_size, num=4):
        self.data = data
        self.X = X
        self.batch_size = batch_size
        self.num = num
    
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        ind = np.random.choice(len(self.X), max(self.batch_size, self.num))
        preds = self.model.predict(self.X[ind], batch_size=self.batch_size)
        preds = preds[:self.num]
        
        for x, y in self.data.batch_to_text((
            self.X[ind[:self.batch_size]], preds)):
                print(' -> '.join((x, y)))
                print('\n{}\n'.format('='*60))
