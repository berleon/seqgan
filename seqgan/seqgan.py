
from keras.engine.training import Model
import numpy as np
import keras.backend as K
from contextlib import contextmanager
import keras.callbacks as cbks

TODO = "todo"


@contextmanager
def trainable(model, trainable):
    trainables = []
    for layer in model.layers:
        trainables.append(layer.trainable)
        layer.trainable = trainable
    yield
    for t, layer in zip(trainables, model.layers):
        layer.trainable = t


class SeqGAN:
    def __init__(self, g, d, m, g_optimizer, d_optimizer):
        self.g = g
        self.d = d
        self.m = m

        self.z, self.seq_input = self.g.inputs
        self.fake_prob = self.g.outputs
        with trainable(m, False):
            self.m_realness = self.m(self.fake_prob)
            self.model_fit_g = Model([self.z, self.seq_input], [self.m_realness])
            self.model_fit_g.compile(g_optimizer, K.binary_crossentropy)

        self.d.compile(d_optimizer, loss=K.binary_crossentropy)

    @property
    def z_shape(self):
        layer, _, _ = self.z._keras_history
        return layer.output_shape

    def sample_z(self):
        shape = self.z_shape
        return np.random.uniform(-1, 1, shape)

    def generate(self, seq_input, z, batch_size=32):
        return self.m.predict([seq_input, z], batch_size=batch_size)

    def train_on_batch(self, seq_input, real, d_target=None):
        if d_target is None:
            d_target = np.concatenate([
                np.zeros((len(seq_input), 1)),
                np.ones((len(real), 1))
            ])
        fake_prob = self.generate(seq_input, self.sample_z())
        fake = np.argmax(fake_prob, axis=TODO)
        d_loss = self.d.train_on_batch(fake, real)
        d_realness = self.d.predict([fake, real], d_target)
        m_loss = self.m.train_on_batch(fake_prob, d_realness)
        g_loss = self.g.train_on_batch(seq_input, self.sample_z())
        return g_loss, d_loss, m_loss

    def fit_generator(self, generator, nb_epoch, nb_batches_per_epoch, callbacks=[],
                      verbose=False):
        out_labels = ['g', 'd', 'm']

        self.history = cbks.History()
        callbacks = [cbks.BaseLogger()] + callbacks + [self.history]
        if verbose:
            callbacks += [cbks.ProgbarLogger()]
        callbacks = cbks.CallbackList(callbacks)
        callbacks._set_model(self)
        callbacks._set_params({
            'nb_epoch': nb_epoch,
            'nb_sample': nb_batches_per_epoch,
            'verbose': verbose,
            'metrics': out_labels,
        })
        callbacks.on_train_begin()

        for e in range(nb_epoch):
            callbacks.on_epoch_begin(e)
            for batch_index, (seq_input, real) in enumerate(generator):
                callbacks.on_batch_begin(batch_index)
                batch_logs = {}
                batch_logs['batch'] = batch_index
                batch_logs['size'] = len(real) + len(seq_input
                                                     )
                outs = self.train_on_batch(seq_input, real)

                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                callbacks.on_batch_end(batch_index, batch_logs)
                if batch_index + 1 == nb_batches_per_epoch:
                    break

            callbacks.on_epoch_end(e)
        callbacks.on_train_end()
