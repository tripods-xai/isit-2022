from dataclasses import replace
from enum import Enum
from typing import Callable, Dict, Union
import tensorflow as tf

from src.channelcoding.codes import Code

from .channelcoding.encoders import AffineConvolutionalCode, GeneralizedConvolutionalCode, TrellisCode
from .channelcoding.bcjr import HazzysTurboDecoder, TurboDecoder
from .dataclasses import SystematicTurboEncoderSpec, NonsystematicTurboEncoderSpec



def turbo_155_7() -> SystematicTurboEncoderSpec:
    code = AffineConvolutionalCode(tf.constant([[1, 1, 1], [1, 0, 1]]), tf.constant([0, 0])).to_rc()
    systematic_code = code.with_systematic()
    interleaved_code = code
    return SystematicTurboEncoderSpec(systematic_code=systematic_code, interleaved_code=interleaved_code)

def turbo_random_7() -> SystematicTurboEncoderSpec:
    code_base = AffineConvolutionalCode(tf.constant([[1, 1, 1], [1, 0, 1]]), tf.constant([0, 0])).to_rc()
    noninterleaved_code = TrellisCode(replace(code_base.trellis, output_table=tf.random.uniform(code_base.trellis.output_table.shape)))
    interleaved_code = TrellisCode(replace(code_base.trellis, output_table=tf.random.uniform(code_base.trellis.output_table.shape)))
    systematic_code = noninterleaved_code.with_systematic()
    return SystematicTurboEncoderSpec(systematic_code=systematic_code, interleaved_code=interleaved_code)

def turbo_755_0() -> NonsystematicTurboEncoderSpec:
    noninterleaved_code = AffineConvolutionalCode(tf.constant([[1, 1, 1], [1, 0, 1]]), tf.constant([0, 0]))
    interleaved_code = AffineConvolutionalCode(tf.constant([[1, 0, 1]]), tf.constant([0]))
    return NonsystematicTurboEncoderSpec(noninterleaved_code=noninterleaved_code, interleaved_code=interleaved_code)

def turboae_approximated_rsc() -> SystematicTurboEncoderSpec:
    systematic_code = AffineConvolutionalCode(tf.constant([[1, 1, 1, 1, 1], [1, 1, 1, 0, 1]]), tf.constant([1, 0])).to_rsc()
    interleaved_code = AffineConvolutionalCode(tf.constant([[1, 1, 1, 1, 1], [0, 1, 1, 1, 1]]), tf.constant([1, 1])).to_rc()
    return SystematicTurboEncoderSpec(systematic_code=systematic_code, interleaved_code=interleaved_code)

def turboae_approximated_nonsys() -> NonsystematicTurboEncoderSpec:
    noninterleaved_code = AffineConvolutionalCode(tf.constant([[1, 1, 1, 1, 1], [1, 1, 1, 0, 1]]), tf.constant([1, 0]))
    interleaved_code = AffineConvolutionalCode(tf.constant([[0, 1, 1, 1, 1]]), tf.constant([1]))
    return NonsystematicTurboEncoderSpec(noninterleaved_code=noninterleaved_code, interleaved_code=interleaved_code)

def turboae_approximated_rsc2() -> SystematicTurboEncoderSpec:
    systematic_code = AffineConvolutionalCode(tf.constant([[1, 1, 1, 0, 1], [1, 1, 1, 1, 1]]), tf.constant([0, 1])).to_rsc()
    interleaved_code = AffineConvolutionalCode(tf.constant([[1, 1, 1, 0, 1], [0, 1, 1, 1, 1]]), tf.constant([0, 1])).to_rc()
    return SystematicTurboEncoderSpec(systematic_code=systematic_code, interleaved_code=interleaved_code)

def turbo_12330_31_rsc() -> SystematicTurboEncoderSpec:
    # --g1 31 --g2 23 --g3 30 --b1 1 --b2 0 --b3 1
    systematic_code = AffineConvolutionalCode(tf.constant([[1, 1, 1, 1, 1], [1, 0, 1, 1, 1]]), tf.constant([1, 0])).to_rsc()
    interleaved_code = AffineConvolutionalCode(tf.constant([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]), tf.constant([1, 1])).to_rc()
    return SystematicTurboEncoderSpec(systematic_code=systematic_code, interleaved_code=interleaved_code)

def turbo_lte() -> SystematicTurboEncoderSpec:
    code = AffineConvolutionalCode(tf.constant([[1, 0, 1, 1], [1, 1, 0, 1]]), tf.constant([0, 0])).to_rc()
    systematic_code = code.with_systematic()
    interleaved_code = code
    return SystematicTurboEncoderSpec(systematic_code=systematic_code, interleaved_code=interleaved_code)

def turbo_random5_1_rsc() -> SystematicTurboEncoderSpec:
    generators = tf.constant(
        [[0, 1, 0, 0, 1],
         [0, 1, 1, 0, 1],
         [0, 1, 1, 0, 1]], dtype=tf.int32)
    biases = tf.constant([1, 0, 1], dtype=tf.int32)
    systematic_code = AffineConvolutionalCode(generators[:2], biases[:2]).to_rsc()
    interleaved_code = AffineConvolutionalCode(tf.gather(generators, [0, 2]), tf.gather(biases, [0, 2])).to_rc()
    return SystematicTurboEncoderSpec(systematic_code=systematic_code, interleaved_code=interleaved_code)

def turbo_random5_2_rsc() -> SystematicTurboEncoderSpec:
    generators = tf.constant(
        [[1, 0, 0, 0, 1],
         [0, 0, 1, 1, 1],
         [1, 0, 0, 1, 1]], dtype=tf.int32)
    biases = tf.constant([0, 1, 0], dtype=tf.int32)
    systematic_code = AffineConvolutionalCode(generators[:2], biases[:2]).to_rsc()
    interleaved_code = AffineConvolutionalCode(tf.gather(generators, [0, 2]), tf.gather(biases, [0, 2])).to_rc()
    return SystematicTurboEncoderSpec(systematic_code=systematic_code, interleaved_code=interleaved_code)

def turbo_random5_3_rsc() -> SystematicTurboEncoderSpec:
    generators = tf.constant(
        [[1, 0, 1, 0, 1],
         [1, 0, 0, 1, 1],
         [1, 0, 0, 1, 1]], dtype=tf.int32)
    biases = tf.constant([0, 1, 0], dtype=tf.int32)
    systematic_code = AffineConvolutionalCode(generators[:2], biases[:2]).to_rsc()
    interleaved_code = AffineConvolutionalCode(tf.gather(generators, [0, 2]), tf.gather(biases, [0, 2])).to_rc()
    return SystematicTurboEncoderSpec(systematic_code=systematic_code, interleaved_code=interleaved_code)

def turbo_random5_4_rsc() -> SystematicTurboEncoderSpec:
    generators = tf.constant(
        [[0, 1, 0, 1, 1],
         [1, 1, 1, 0, 1],
         [0, 0, 0, 1, 1]], dtype=tf.int32)
    biases = tf.constant([0, 0, 0], dtype=tf.int32)
    systematic_code = AffineConvolutionalCode(generators[:2], biases[:2]).to_rsc()
    interleaved_code = AffineConvolutionalCode(tf.gather(generators, [0, 2]), tf.gather(biases, [0, 2])).to_rc()
    return SystematicTurboEncoderSpec(systematic_code=systematic_code, interleaved_code=interleaved_code)

def turbo_random5_5_rsc() -> SystematicTurboEncoderSpec:
    generators = tf.constant(
        [[1, 1, 1, 1, 1],
         [1, 0, 1, 1, 1],
         [0, 0, 1, 1, 1]], dtype=tf.int32)
    biases = tf.constant([1, 0, 0], dtype=tf.int32)
    systematic_code = AffineConvolutionalCode(generators[:2], biases[:2]).to_rsc()
    interleaved_code = AffineConvolutionalCode(tf.gather(generators, [0, 2]), tf.gather(biases, [0, 2])).to_rc()
    return SystematicTurboEncoderSpec(systematic_code=systematic_code, interleaved_code=interleaved_code)


def turbo_random5_1_nonsys() -> NonsystematicTurboEncoderSpec:
    generators = tf.constant(
        [[0, 1, 0, 0, 1],
         [0, 1, 1, 0, 1],
         [0, 1, 1, 0, 1]], dtype=tf.int32)
    biases = tf.constant([1, 0, 1], dtype=tf.int32)
    noninterleaved_code = AffineConvolutionalCode(generators[:2], biases[:2])
    interleaved_code = AffineConvolutionalCode(generators[2:3], biases[2:3])
    return NonsystematicTurboEncoderSpec(noninterleaved_code=noninterleaved_code, interleaved_code=interleaved_code)

def turbo_random5_2_nonsys() -> NonsystematicTurboEncoderSpec:
    generators = tf.constant(
        [[1, 0, 0, 0, 1],
         [0, 0, 1, 1, 1],
         [1, 0, 0, 1, 1]], dtype=tf.int32)
    biases = tf.constant([0, 1, 0], dtype=tf.int32)
    noninterleaved_code = AffineConvolutionalCode(generators[:2], biases[:2])
    interleaved_code = AffineConvolutionalCode(generators[2:3], biases[2:3])
    return NonsystematicTurboEncoderSpec(noninterleaved_code=noninterleaved_code, interleaved_code=interleaved_code)

def turbo_random5_3_nonsys() -> NonsystematicTurboEncoderSpec:
    generators = tf.constant(
        [[1, 0, 1, 0, 1],
         [1, 0, 0, 1, 1],
         [1, 0, 0, 1, 1]], dtype=tf.int32)
    biases = tf.constant([0, 1, 0], dtype=tf.int32)
    noninterleaved_code = AffineConvolutionalCode(generators[:2], biases[:2])
    interleaved_code = AffineConvolutionalCode(generators[2:3], biases[2:3])
    return NonsystematicTurboEncoderSpec(noninterleaved_code=noninterleaved_code, interleaved_code=interleaved_code)

def turbo_random5_4_nonsys() -> NonsystematicTurboEncoderSpec:
    generators = tf.constant(
        [[0, 1, 0, 1, 1],
         [1, 1, 1, 0, 1],
         [0, 0, 0, 1, 1]], dtype=tf.int32)
    biases = tf.constant([0, 0, 0], dtype=tf.int32)
    noninterleaved_code = AffineConvolutionalCode(generators[:2], biases[:2])
    interleaved_code = AffineConvolutionalCode(generators[2:3], biases[2:3])
    return NonsystematicTurboEncoderSpec(noninterleaved_code=noninterleaved_code, interleaved_code=interleaved_code)

def turbo_random5_5_nonsys() -> NonsystematicTurboEncoderSpec:
    generators = tf.constant(
        [[1, 1, 1, 1, 1],
         [1, 0, 1, 1, 1],
         [0, 0, 1, 1, 1]], dtype=tf.int32)
    biases = tf.constant([1, 0, 0], dtype=tf.int32)
    noninterleaved_code = AffineConvolutionalCode(generators[:2], biases[:2])
    interleaved_code = AffineConvolutionalCode(generators[2:3], biases[2:3])
    return NonsystematicTurboEncoderSpec(noninterleaved_code=noninterleaved_code, interleaved_code=interleaved_code)


TURBOAE_EXACT_TABLE1 = tf.constant(
    [[1,0],
     [0,1],
     [0,0],
     [1,1],
     [1,1],
     [0,0],
     [1,1],
     [0,0],
     [0,1],
     [1,0],
     [1,1],
     [0,0],
     [1,0],
     [0,1],
     [0,0],
     [1,1],
     [0,1],
     [1,0],
     [1,1],
     [0,0],
     [0,0],
     [0,1],
     [0,0],
     [1,1],
     [1,0],
     [0,1],
     [0,0],
     [1,1],
     [0,1],
     [1,0],
     [1,1],
     [0,0]], dtype=tf.float32
)
TURBOAE_EXACT_TABLE2 = tf.constant(
    [[1],
     [0],
     [0],
     [1],
     [0],
     [1],
     [1],
     [0],
     [0],
     [1],
     [1],
     [0],
     [1],
     [0],
     [0],
     [1],
     [0],
     [1],
     [1],
     [0],
     [0],
     [1],
     [1],
     [0],
     [1],
     [0],
     [0],
     [1],
     [1],
     [0],
     [0],
     [1]], dtype=tf.float32
)

def turboae_binary_exact_nonsys() -> NonsystematicTurboEncoderSpec:
    noninterleaved_code = GeneralizedConvolutionalCode(TURBOAE_EXACT_TABLE1)
    interleaved_code = GeneralizedConvolutionalCode(TURBOAE_EXACT_TABLE2)
    return NonsystematicTurboEncoderSpec(noninterleaved_code=noninterleaved_code, interleaved_code=interleaved_code)

def turboae_binary_exact_rsc() -> SystematicTurboEncoderSpec:
    # Swap code 1 and 2 since only 2 can be inverted
    swapped_table = tf.stack([TURBOAE_EXACT_TABLE1[:, 1], TURBOAE_EXACT_TABLE1[:, 0]], axis=1)
    systematic_code = GeneralizedConvolutionalCode(swapped_table).to_rsc()
    interleaved_table = tf.stack([TURBOAE_EXACT_TABLE1[:, 1], TURBOAE_EXACT_TABLE2[:, 0]], axis=1)
    interleaved_code = GeneralizedConvolutionalCode(interleaved_table).to_rc()
    return SystematicTurboEncoderSpec(systematic_code=systematic_code, interleaved_code=interleaved_code)

class SystematicEncoders(Enum):
    TURBO_155_7 = 'turbo-155-7'
    TURBOAE_BINARY_EXACT_RSC = 'turboae-binary-exact-rsc'
    TURBOAE_APPROXIMATED_RSC = 'turboae-approximated-rsc'
    TURBO_12330_31_RSC = 'turbo-12330-31-rsc'
    TURBO_LTE = 'turbo-lte'
    TURBO_RANDOM_7 = 'turbo-random-7'
    TURBO_RANDOM5_1_RSC = 'turbo-random5-1-rsc'
    TURBO_RANDOM5_2_RSC = 'turbo-random5-2-rsc'
    TURBO_RANDOM5_3_RSC = 'turbo-random5-3-rsc'
    TURBO_RANDOM5_4_RSC = 'turbo-random5-4-rsc'
    TURBO_RANDOM5_5_RSC = 'turbo-random5-5-rsc'
    TURBOAE_APPROXIMATED_RSC2 = 'turboae-approximated-rsc2'

    def __str__(self):
        return self.value

class NonsystematicEncoders(Enum):
    TURBO_755_0 = 'turbo-755-0'
    TURBOAE_BINARY_EXACT = 'turboae-binary-exact'
    TURBOAE_APPROXIMATED_NONSYS = 'turboae-approximated-nonsys'
    TURBO_RANDOM5_1_NONSYS = 'turbo-random5-1-nonsys'
    TURBO_RANDOM5_2_NONSYS = 'turbo-random5-2-nonsys'
    TURBO_RANDOM5_3_NONSYS = 'turbo-random5-3-nonsys'
    TURBO_RANDOM5_4_NONSYS = 'turbo-random5-4-nonsys'
    TURBO_RANDOM5_5_NONSYS = 'turbo-random5-5-nonsys'

    def __str__(self):
        return self.value

ENCODERS: Dict[Enum, Callable[[], Union[SystematicTurboEncoderSpec, NonsystematicTurboEncoderSpec]]] = {
    SystematicEncoders.TURBO_155_7: turbo_155_7,
    NonsystematicEncoders.TURBO_755_0: turbo_755_0,
    SystematicEncoders.TURBOAE_BINARY_EXACT_RSC: turboae_binary_exact_rsc,
    NonsystematicEncoders.TURBOAE_BINARY_EXACT: turboae_binary_exact_nonsys,
    SystematicEncoders.TURBOAE_APPROXIMATED_RSC: turboae_approximated_rsc,
    NonsystematicEncoders.TURBOAE_APPROXIMATED_NONSYS: turboae_approximated_nonsys,
    SystematicEncoders.TURBO_12330_31_RSC: turbo_12330_31_rsc,
    SystematicEncoders.TURBO_LTE: turbo_lte,
    SystematicEncoders.TURBO_RANDOM_7: turbo_random_7,
    SystematicEncoders.TURBO_RANDOM5_1_RSC: turbo_random5_1_rsc,
    SystematicEncoders.TURBO_RANDOM5_2_RSC: turbo_random5_2_rsc,
    SystematicEncoders.TURBO_RANDOM5_3_RSC: turbo_random5_3_rsc,
    SystematicEncoders.TURBO_RANDOM5_4_RSC: turbo_random5_4_rsc,
    SystematicEncoders.TURBO_RANDOM5_5_RSC: turbo_random5_5_rsc,
    NonsystematicEncoders.TURBO_RANDOM5_1_NONSYS: turbo_random5_1_nonsys,
    NonsystematicEncoders.TURBO_RANDOM5_2_NONSYS: turbo_random5_2_nonsys,
    NonsystematicEncoders.TURBO_RANDOM5_3_NONSYS: turbo_random5_3_nonsys,
    NonsystematicEncoders.TURBO_RANDOM5_4_NONSYS: turbo_random5_4_nonsys,
    NonsystematicEncoders.TURBO_RANDOM5_5_NONSYS: turbo_random5_5_nonsys,
    SystematicEncoders.TURBOAE_APPROXIMATED_RSC2: turboae_approximated_rsc2,
}

class SystematicDecoders(Enum):
    HAZZYS = 'hazzys'
    BASIC = 'basic'

    def __str__(self):
        return self.value

class NonsystematicDecoders(Enum):
    BASIC = 'basic'

    def __str__(self):
        return self.value

DECODERS = {
    SystematicDecoders.HAZZYS: HazzysTurboDecoder,
    SystematicDecoders.BASIC: TurboDecoder,
    NonsystematicDecoders.BASIC: TurboDecoder
}

