from keras.engine.topology import Input
import numpy as np


TODO = "todo"


class SeqGAN:
    def __init__(self, g, d, m):
        self.g = g
        self.d = d
        self.m = m

        self.seq_input = Input()
        self.z = Input()
        self.real = Input()
        self.fake_prob = self.g(self.seq_input)

    def sample_z(self):
        pass

    def generate(self, seq_input, z):
        pass

    def predict_d_realness(self):
        pass

    def fit_d(self, fake, real):
        pass

    def fit_m(self, fake_prob, d_realness):
        pass

    def fit_g(self, seq_input):
        pass

    def fit_generator(self, generator, nb_epochs, nb_batches_per_epoch):
        for e in range(nb_epochs):
            for i, (seq_input, real) in enumerate(generator):
                fake_prob = self.generate(seq_input, self.sample_z())
                fake = np.argmax(fake_prob, axis=TODO)
                d_realness = self.fit_d(fake, real)
                self.fit_m(fake_prob, d_realness)
                self.fit_g(seq_input)
                if i + 1 == nb_batches_per_epoch:
                    break
