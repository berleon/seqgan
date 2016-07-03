from keras.utils.data_utils import get_file
import numpy as np
import random
import sys


class TextSequenceData:
    def __init__(self, fname, origin, inlen=100, outlen=50, step=10):
        self.inlen = inlen
        self.outlen = outlen
        self.step = step

        self.path = get_file(fname, origin=origin)
        text = open(self.path, encoding="utf-8").read().lower()

        self.chars = sorted(list(set(text)))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

        self.build_dataset(text)


    def build_dataset(self, text):
        insequences = []
        outsequences = []
        for i in range(0, len(text) - self.inlen - self.outlen, self.step):
            iout = i + self.inlen
            insequences.append(text[i:iout])
            outsequences.append(text[iout:iout+self.outlen])

        self.X = np.zeros((len(insequences), self.inlen, len(self.chars)), 
                          dtype=np.bool)
        self.Y = np.zeros((len(outsequences), self.outlen, len(self.chars)), 
                          dtype=np.bool)
        for i, seq in enumerate(insequences):
            for t, char in enumerate(seq):
                self.X[i, t, self.char_indices[char]] = True
        for i, seq in enumerate(outsequences):
            for t, char in enumerate(seq):
                self.Y[i, t, self.char_indices[char]] = True

    def seq_to_text(self, seq):
        chars = []
        for char_arr in seq:
            ind = np.where(char_arr)[0][0]
            chars.append(self.indices_char[ind])
        return ''.join(chars)

    def batch_to_text(self, batch):
        X, Y = batch
        return zip([self.seq_to_text(s) for s in X],
                   [self.seq_to_text(s) for s in Y])



