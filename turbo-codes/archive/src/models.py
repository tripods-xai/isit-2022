import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

from .encoders import SystematicConvEncoder, TurboSystematicSharedEncoder, CNNTrellisHelper, TurboSystematicSeparateEncoder, NonsystematicConvEncoder, TurboNonsystematicSharedEncoder
from .constraints import PowerConstraint
from .channels import TFAWGN
from .tensorflow_turbo import turbo_decode
from .utils import snr2sigma
from .interleavers import make_tf_interleaver



class TurboEncoderDecoder(layers.Layer):
    def __init__(self, turbo_encoder, sigma, num_decode_iter):
        super().__init__()
        self.turbo_encoder = turbo_encoder
        self.num_decode_iter = num_decode_iter
        self.sigma = sigma
        self.noisy_channel = TFAWGN(sigma)

    def call(self, x):
        msg_object = self.turbo_encoder(x, self.noisy_channel)
        L_int = tf.zeros_like(msg_object["stream1"])
        logits = turbo_decode(
            msg_object["stream1"], 
            msg_object["stream2"], 
            msg_object["stream1_i"], 
            msg_object["stream3"], 
            msg_object["trellis1"], 
            msg_object["trellis2"], 
            msg_object["interleaver"], 
            L_int, 
            self.sigma, 
            num_iter=self.num_decode_iter,
            use_max=True
        )
        return logits
    
def syscnn1_1000_2m_shared(snr=0.0, block_len=100, num_decode_iter=10, batch_size=1000):
    kernel_dims = (3,1,)
    encoder = SystematicConvEncoder(n_outputs=3, hidden_layers=1, hidden_dim=1000, kernel_dims=kernel_dims)
    print(encoder.get_config())
    constraint = PowerConstraint(center=0., scale=1.)
    trellis_helper = CNNTrellisHelper(encoder.window)
    tf_interleaver = make_tf_interleaver(block_len)
    turbo_encoder = TurboSystematicSharedEncoder(encoder, constraint, trellis_helper, tf_interleaver)
    sigma = snr2sigma(snr)

    encoder_decoder = TurboEncoderDecoder(turbo_encoder, sigma, num_decode_iter)

    inputs = keras.Input(shape=(block_len,), batch_size=batch_size)
    outputs = encoder_decoder(inputs)
    return keras.Model(inputs=inputs, outputs=outputs)

def syscnn1_1000_4m_shared(snr=0.0, block_len=100, num_decode_iter=10, batch_size=1000):
    kernel_dims = (5,1,)
    encoder = SystematicConvEncoder(n_outputs=3, hidden_layers=1, hidden_dim=1000, kernel_dims=kernel_dims)
    print(encoder.get_config())
    constraint = PowerConstraint(center=0., scale=1.)
    trellis_helper = CNNTrellisHelper(encoder.window)
    tf_interleaver = make_tf_interleaver(block_len)
    turbo_encoder = TurboSystematicSharedEncoder(encoder, constraint, trellis_helper, tf_interleaver)
    sigma = snr2sigma(snr)

    encoder_decoder = TurboEncoderDecoder(turbo_encoder, sigma, num_decode_iter)

    inputs = keras.Input(shape=(block_len,), batch_size=batch_size)
    outputs = encoder_decoder(inputs)
    return keras.Model(inputs=inputs, outputs=outputs)


def syscnn0_2m_shared(snr=0.0, block_len=100, num_decode_iter=10, batch_size=1000):
    kernel_dims = (3,1,)
    encoder = SystematicConvEncoder(n_outputs=3, hidden_layers=0, hidden_dim=1, kernel_dims=kernel_dims)
    print(encoder.get_config())
    constraint = PowerConstraint(center=0., scale=1.)
    trellis_helper = CNNTrellisHelper(encoder.window)
    tf_interleaver = make_tf_interleaver(block_len)
    turbo_encoder = TurboSystematicSharedEncoder(encoder, constraint, trellis_helper, tf_interleaver)
    sigma = snr2sigma(snr)

    encoder_decoder = TurboEncoderDecoder(turbo_encoder, sigma, num_decode_iter)

    inputs = keras.Input(shape=(block_len,), batch_size=batch_size)
    outputs = encoder_decoder(inputs)
    return keras.Model(inputs=inputs, outputs=outputs)

def syscnn1_1000_4m_separate(snr=0.0, block_len=100, num_decode_iter=10, batch_size=1000):
    kernel_dims = (5,1,)
    encoder1 = SystematicConvEncoder(n_outputs=2, hidden_layers=1, hidden_dim=1000, kernel_dims=kernel_dims)
    encoder2 = SystematicConvEncoder(n_outputs=2, hidden_layers=1, hidden_dim=1000, kernel_dims=kernel_dims)
    print(encoder1.get_config())
    print(encoder2.get_config())
    constraint1 = PowerConstraint(center=0., scale=1.)
    constraint2 = PowerConstraint(center=0., scale=1.)
    trellis_helper = CNNTrellisHelper(encoder.window)
    tf_interleaver = make_tf_interleaver(block_len)
    turbo_encoder = TurboSystematicSeparateEncoder(encoder1, encoder2, constraint1, constraint2, trellis_helper, tf_interleaver)
    sigma = snr2sigma(snr)

    encoder_decoder = TurboEncoderDecoder(turbo_encoder, sigma, num_decode_iter)

    inputs = keras.Input(shape=(block_len,), batch_size=batch_size)
    outputs = encoder_decoder(inputs)
    return keras.Model(inputs=inputs, outputs=outputs)

def syscnn2_1000_33_5m_shared(snr=0.0, block_len=100, num_decode_iter=10, batch_size=1000):
    kernel_dims = (3,3,1)
    encoder = SystematicConvEncoder(n_outputs=3, hidden_layers=2, hidden_dim=1000, kernel_dims=kernel_dims)
    print(encoder.get_config())
    constraint = PowerConstraint(center=0., scale=1.)
    trellis_helper = CNNTrellisHelper(encoder.window)
    tf_interleaver = make_tf_interleaver(block_len)
    turbo_encoder = TurboSystematicSharedEncoder(encoder, constraint, trellis_helper, tf_interleaver)
    sigma = snr2sigma(snr)

    encoder_decoder = TurboEncoderDecoder(turbo_encoder, sigma, num_decode_iter)

    inputs = keras.Input(shape=(block_len,), batch_size=batch_size)
    outputs = encoder_decoder(inputs)
    return keras.Model(inputs=inputs, outputs=outputs)

def syscnn2_708_33_5m_separate(snr=0.0, block_len=100, num_decode_iter=10, batch_size=500):
    kernel_dims = (3,3,1)
    encoder1 = SystematicConvEncoder(n_outputs=2, hidden_layers=2, hidden_dim=708, kernel_dims=kernel_dims)
    encoder2 = SystematicConvEncoder(n_outputs=2, hidden_layers=2, hidden_dim=708, kernel_dims=kernel_dims)
    assert encoder1.window == encoder2.window
    print(encoder1.get_config())
    print(encoder2.get_config())
    constraint1 = PowerConstraint(center=0., scale=1.)
    constraint2 = PowerConstraint(center=0., scale=1.)
    trellis_helper = CNNTrellisHelper(encoder1.window)
    tf_interleaver = make_tf_interleaver(block_len)
    turbo_encoder = TurboSystematicSeparateEncoder(encoder1, encoder2, constraint1, constraint2, trellis_helper, tf_interleaver)
    sigma = snr2sigma(snr)

    encoder_decoder = TurboEncoderDecoder(turbo_encoder, sigma, num_decode_iter)

    inputs = keras.Input(shape=(block_len,), batch_size=batch_size)
    outputs = encoder_decoder(inputs)
    return keras.Model(inputs=inputs, outputs=outputs)

def syscnn3_1000_333_7m_shared(snr=0.0, block_len=100, num_decode_iter=10, batch_size=1000):
    kernel_dims = (3,3,3,1)
    encoder = SystematicConvEncoder(n_outputs=3, hidden_layers=3, hidden_dim=1000, kernel_dims=kernel_dims)
    print(encoder.get_config())
    constraint = PowerConstraint(center=0., scale=1.)
    trellis_helper = CNNTrellisHelper(encoder.window)
    tf_interleaver = make_tf_interleaver(block_len)
    turbo_encoder = TurboSystematicSharedEncoder(encoder, constraint, trellis_helper, tf_interleaver)
    sigma = snr2sigma(snr)

    encoder_decoder = TurboEncoderDecoder(turbo_encoder, sigma, num_decode_iter)

    inputs = keras.Input(shape=(block_len,), batch_size=batch_size)
    outputs = encoder_decoder(inputs)
    return keras.Model(inputs=inputs, outputs=outputs)

def syscnn3_1000_551_9m_shared(snr=0.0, block_len=100, num_decode_iter=10, batch_size=1000):
    kernel_dims = (5,5,1)
    encoder = SystematicConvEncoder(n_outputs=3, hidden_layers=2, hidden_dim=1000, kernel_dims=kernel_dims)
    print(encoder.get_config())
    constraint = PowerConstraint(center=0., scale=1.)
    trellis_helper = CNNTrellisHelper(encoder.window)
    tf_interleaver = make_tf_interleaver(block_len)
    turbo_encoder = TurboSystematicSharedEncoder(encoder, constraint, trellis_helper, tf_interleaver)
    sigma = snr2sigma(snr)

    encoder_decoder = TurboEncoderDecoder(turbo_encoder, sigma, num_decode_iter)

    inputs = keras.Input(shape=(block_len,), batch_size=batch_size)
    outputs = encoder_decoder(inputs)
    return keras.Model(inputs=inputs, outputs=outputs)
    
def syscnn2_1000_3311_5m_shared(snr=0.0, block_len=100, num_decode_iter=10, batch_size=1000):
    kernel_dims = (3,3,1, 1)
    encoder = SystematicConvEncoder(n_outputs=3, hidden_layers=3, hidden_dim=1000, kernel_dims=kernel_dims)
    print(encoder.get_config())
    constraint = PowerConstraint(center=0., scale=1.)
    trellis_helper = CNNTrellisHelper(encoder.window)
    tf_interleaver = make_tf_interleaver(block_len)
    turbo_encoder = TurboSystematicSharedEncoder(encoder, constraint, trellis_helper, tf_interleaver)
    sigma = snr2sigma(snr)

    encoder_decoder = TurboEncoderDecoder(turbo_encoder, sigma, num_decode_iter)

    inputs = keras.Input(shape=(block_len,), batch_size=batch_size)
    outputs = encoder_decoder(inputs)
    return keras.Model(inputs=inputs, outputs=outputs)

def syscnn3_1000_33331_9m_shared(snr=0.0, block_len=100, num_decode_iter=10, batch_size=1000):
    kernel_dims = (3,3,3,3,1)
    encoder = SystematicConvEncoder(n_outputs=3, hidden_layers=4, hidden_dim=1000, kernel_dims=kernel_dims)
    print(encoder.get_config())
    constraint = PowerConstraint(center=0., scale=1.)
    trellis_helper = CNNTrellisHelper(encoder.window)
    tf_interleaver = make_tf_interleaver(block_len)
    turbo_encoder = TurboSystematicSharedEncoder(encoder, constraint, trellis_helper, tf_interleaver)
    sigma = snr2sigma(snr)

    encoder_decoder = TurboEncoderDecoder(turbo_encoder, sigma, num_decode_iter)

    inputs = keras.Input(shape=(block_len,), batch_size=batch_size)
    outputs = encoder_decoder(inputs)
    return keras.Model(inputs=inputs, outputs=outputs)

def syscnn4_1000_22221_5m_shared(snr=0.0, block_len=100, num_decode_iter=10, batch_size=1000):
    kernel_dims = (2,2,2,2,1)
    encoder = SystematicConvEncoder(n_outputs=3, hidden_layers=4, hidden_dim=1000, kernel_dims=kernel_dims)
    print(encoder.get_config())
    constraint = PowerConstraint(center=0., scale=1.)
    trellis_helper = CNNTrellisHelper(encoder.window)
    tf_interleaver = make_tf_interleaver(block_len)
    turbo_encoder = TurboSystematicSharedEncoder(encoder, constraint, trellis_helper, tf_interleaver)
    sigma = snr2sigma(snr)

    encoder_decoder = TurboEncoderDecoder(turbo_encoder, sigma, num_decode_iter)

    inputs = keras.Input(shape=(block_len,), batch_size=batch_size)
    outputs = encoder_decoder(inputs)
    return keras.Model(inputs=inputs, outputs=outputs)

def nonsyscnn4_1000_22221_5m_shared(snr=0.0, block_len=100, num_decode_iter=10, batch_size=1000):
    kernel_dims = (2,2,2,2,1)
    encoder = NonsystematicConvEncoder(n_outputs=3, hidden_layers=4, hidden_dim=1000, kernel_dims=kernel_dims)
    print(encoder.get_config())
    constraint = PowerConstraint(center=0., scale=1.)
    trellis_helper = CNNTrellisHelper(encoder.window)
    tf_interleaver = make_tf_interleaver(block_len)
    turbo_encoder = TurboNonsystematicSharedEncoder(encoder, constraint, trellis_helper, tf_interleaver)
    sigma = snr2sigma(snr)

    encoder_decoder = TurboEncoderDecoder(turbo_encoder, sigma, num_decode_iter)

    inputs = keras.Input(shape=(block_len,), batch_size=batch_size)
    outputs = encoder_decoder(inputs)
    return keras.Model(inputs=inputs, outputs=outputs)