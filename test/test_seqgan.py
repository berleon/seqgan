from seqgan.seqgan import SeqGAN
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.engine.training import Model
from keras.optimizers import Adam

from keras.engine.topology import Input, merge
from beras.util import sequential


def test_seqgan():
    output_size = 64
    input_size = 64
    z_size = 48
    nb_chars = 32

    seq = Input(shape=(input_size, nb_chars))
    z = Input(shape=(z_size,))
    z_rep = RepeatVector(input_size)(z)
    seq_and_z = merge([seq, z_rep], mode='concat', concat_axis=-1)
    fake_prob = sequential([
        LSTM(8),
        RepeatVector(output_size),
        LSTM(8, return_sequences=True),
        TimeDistributed(Dense(nb_chars, activation='softmax')),
    ])(seq_and_z)

    g = Model([z, seq], [fake_prob])

    x = Input(shape=(input_size + output_size, nb_chars))
    d_realness = sequential([
        LSTM(8),
        Dense(1, activation='sigmoid'),
    ])(x)
    d = Model([x], [d_realness])

    m_realness = sequential([
        LSTM(8),
        Dense(1, activation='sigmoid'),
    ])(x)
    m = Model([x], [m_realness])
    m.compile(Adam(), 'mse')
    gan = SeqGAN(g, d, m, Adam(), Adam())
    assert gan.z_shape == (None, z_size)
