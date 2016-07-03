import string
import random

from seq2seq.layers.encoders import LSTMEncoder
from seq2seq.layers.state_transfer_rnn import StateTransferLSTM

import keras.backend as K
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.engine.topology import InputSpec


class LSTMDecoder(StateTransferLSTM):
    '''
    A basic LSTM decoder. Similar to [1].
    The output of at each timestep is the input to the next timestep.
    The input to the first timestep is the context vector from the encoder.

    Basic equation:
        y(t) = LSTM(s(t-1), y(t-1)); Where s is the hidden state of the LSTM (h and c)
        y(0) = LSTM(s0, C); C is the context vector from the encoder.

    In addition, the hidden state of the encoder is usually used to initialize the hidden
    state of the decoder. Checkout models.py to see how its done.
    '''
    input_ndim = 2

    def __init__(self, output_length, hidden_dim=None, **kwargs):
        self.output_length = output_length
        self.hidden_dim = hidden_dim
        input_dim = None
        if 'input_dim' in kwargs:
            kwargs['output_dim'] = input_dim
        if 'input_shape' in kwargs:
            kwargs['output_dim'] = kwargs['input_shape'][-1]
        elif 'batch_input_shape' in kwargs:
            kwargs['output_dim'] = kwargs['batch_input_shape'][-1]
        elif 'output_dim' not in kwargs:
            kwargs['output_dim'] = None
        super(LSTMDecoder, self).__init__(**kwargs)
        self.return_sequences = True
        self.updates = []
        self.consume_less = 'mem'

    def build(self, input_shape):
        input_shape = list(input_shape)
        input_shape = input_shape[:1] + [self.output_length] + input_shape[1:]
        if not self.hidden_dim:
            self.hidden_dim = input_shape[-1]
        output_dim = input_shape[-1]
        self.output_dim = self.hidden_dim
        initial_weights = self.initial_weights
        self.initial_weights = None
        super(LSTMDecoder, self).build(input_shape)
        self.output_dim = output_dim
        self.initial_weights = initial_weights
        rand_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
        self.W_y = self.init((self.hidden_dim, self.output_dim), name=rand_name)
        rand_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
        self.b_y = K.zeros((self.output_dim), name=rand_name)
        self.trainable_weights += [self.W_y, self.b_y]
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        input_shape.pop(1)
        self.input_spec = [InputSpec(shape=tuple(input_shape))]

    def get_constants(self, x):
        output_dim = self.output_dim
        self.output_dim = self.hidden_dim
        consts = super(LSTMDecoder, self).get_constants(x)
        self.output_dim = output_dim
        return consts

    def reset_states(self):
        output_dim = self.output_dim
        self.output_dim = self.hidden_dim
        super(LSTMDecoder, self).reset_states()
        self.output_dim = output_dim

    def get_initial_states(self, x):
        output_dim = self.output_dim
        self.output_dim = self.hidden_dim
        initial_states = super(LSTMDecoder, self).get_initial_states(x)
        self.output_dim = output_dim
        return initial_states

    def step(self, x, states):
        assert len(states) == 5, len(states)
        states = list(states)
        y_tm1 = states.pop(2)
        output_dim = self.output_dim
        self.output_dim = self.hidden_dim
        h_t, new_states = super(LSTMDecoder, self).step(y_tm1, states)
        self.output_dim = output_dim
        y_t = self.activation(K.dot(h_t, self.W_y) + self.b_y)
        new_states += [y_t]
        return y_t, new_states

    def call(self, x, mask=None):
        X = K.repeat(x, self.output_length)
        input_shape = list(self.input_spec[0].shape)
        input_shape = input_shape[:1] + [self.output_length] + input_shape[1:]
        self.input_spec = [InputSpec(shape=tuple(input_shape))]
        if self.stateful or self.state_input or len(self.state_outputs) > 0:
            initial_states = self.states[:]
        else:
            initial_states = self.get_initial_states(X)
        constants = self.get_constants(X)
        y_0 = K.permute_dimensions(X, (1, 0, 2))[0]
        initial_states += [y_0]
        last_output, outputs, states = K.rnn(self.step, X,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=self.output_length)
        if self.stateful and not self.state_input:
            self.updates = []
            for i in range(2):
                self.updates.append((self.states[i], states[i]))
        self.states_to_transfer = states
        input_shape.pop(1)
        self.input_spec = [InputSpec(shape=input_shape)]
        return outputs

    def assert_input_compatibility(self, x):
        shape = x._keras_shape
        assert K.ndim(x) == 2, "LSTMDecoder requires 2D  input, not " + str(K.ndim(x)) + "D."
        assert shape[-1] == self.output_dim or not self.output_dim, \
            "output_dim of LSTMDecoder should be same as the last dimension in" \
            "the input shape. output_dim = " + str(self.output_dim) + \
            ", got tensor with shape : " + str(shape) + "."

    def get_output_shape_for(self, input_shape):
        input_shape = list(input_shape)
        output_shape = input_shape[:1] + [self.output_length] + input_shape[1:]
        return tuple(output_shape)

    def get_config(self):
        config = {
            'name': self.__class__.__name__,
            'hidden_dim': self.hidden_dim,
            'output_length': self.output_length
        }
        base_config = super(LSTMDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Seq2seq(Sequential):
        '''
        Seq2seq model based on [1] and [2].
        This model has the ability to transfer the encoder hidden state to the decoder's
        hidden state(specified by the broadcast_state argument). Also, in deep models
        (depth > 1), the hidden state is propogated throughout the LSTM stack(specified by
        the inner_broadcast_state argument. You can switch between [1] based model and [2]
        based model using the peek argument.(peek = True for [2], peek = False for [1]).
        When peek = True, the decoder gets a 'peek' at the context vector at every timestep.

        [1] based model:

                Encoder:
                X = Input sequence
                C = LSTM(X); The context vector

                Decoder:
        y(t) = LSTM(s(t-1), y(t-1)); Where s is the hidden state of the LSTM (h and c)
        y(0) = LSTM(s0, C); C is the context vector from the encoder.

    [2] based model:

                Encoder:
                X = Input sequence
                C = LSTM(X); The context vector

                Decoder:
        y(t) = LSTM(s(t-1), y(t-1), C)
        y(0) = LSTM(s0, C, C)
        Where s is the hidden state of the LSTM (h and c), and C is the context vector
        from the encoder.

        Arguments:

        output_dim : Required output dimension.
        hidden_dim : The dimension of the internal representations of the model.
        output_length : Length of the required output sequence.
        depth : Used to create a deep Seq2seq model. For example, if depth = 3,
                        there will be 3 LSTMs on the enoding side and 3 LSTMs on the
                        decoding side. You can also specify depth as a tuple. For example,
                        if depth = (4, 5), 4 LSTMs will be added to the encoding side and
                        5 LSTMs will be added to the decoding side.
        broadcast_state : Specifies whether the hidden state from encoder should be
                                        transfered to the deocder.
        inner_broadcast_state : Specifies whether hidden states should be propogated
                                                        throughout the LSTM stack in deep models.
        peek : Specifies if the decoder should be able to peek at the context vector
                at every timestep.
        dropout : Dropout probability in between layers.


        '''
        def __init__(self, output_dim, hidden_dim, output_length, depth=1, broadcast_state=True,
                     inner_broadcast_state=True, peek=False, dropout=0.1, **kwargs):
                super(Seq2seq, self).__init__()
                if type(depth) not in [list, tuple]:
                        depth = (depth, depth)
                if 'batch_input_shape' in kwargs:
                        shape = kwargs['batch_input_shape']
                        del kwargs['batch_input_shape']
                elif 'input_shape' in kwargs:
                        shape = (None,) + tuple(kwargs['input_shape'])
                        del kwargs['input_shape']
                elif 'input_dim' in kwargs:
                        shape = (None, None, kwargs['input_dim'])
                        del kwargs['input_dim']
                lstms = []
                layer = LSTMEncoder(batch_input_shape=shape, output_dim=hidden_dim,
                                    state_input=False, return_sequences=depth[0] > 1, **kwargs)
                self.add(layer)
                lstms += [layer]
                for i in range(depth[0] - 1):
                        self.add(Dropout(dropout))
                        layer = LSTMEncoder(output_dim=hidden_dim,
                                            state_input=inner_broadcast_state,
                                            return_sequences=i < depth[0] - 2, **kwargs)
                        self.add(layer)
                        lstms += [layer]
                if inner_broadcast_state:
                        for i in range(len(lstms) - 1):
                                lstms[i].broadcast_state(lstms[i + 1])
                encoder = self.layers[-1]
                self.add(Dropout(dropout))
                decoder = LSTMDecoder(hidden_dim=hidden_dim, output_length=output_length,
                                      state_input=broadcast_state, **kwargs)

                self.add(decoder)
                lstms = [decoder]

                for i in range(depth[1] - 1):
                        self.add(Dropout(dropout))
                        layer = LSTMEncoder(
                            output_dim=hidden_dim, state_input=inner_broadcast_state,
                            return_sequences=True, **kwargs)
                        self.add(layer)
                        lstms += [layer]
                        self.add(Dropout(dropout))

                if inner_broadcast_state:
                                for i in range(len(lstms) - 1):
                                        lstms[i].broadcast_state(lstms[i + 1])
                if broadcast_state:
                        encoder.broadcast_state(decoder)
                self.add(Dropout(dropout))
                self.add(TimeDistributed(Dense(output_dim, activation='softmax')))
                self.encoder = encoder
                self.decoder = decoder
