__author__ = 'abhmul'
'''
Evaluate
'''
from typing import cast
import argparse
import tensorflow as tf

from src.channelcoding.channels import AWGN, AdditiveTonAWGN, NonIIDMarkovianGaussianAsAWGN, identity_constraint, scale_constraint
from src.channelcoding.encoder_decoders import TurboSystematicEncoderDecoder
from src.channelcoding.losses import cross_entropy_with_logits
from src.codes import DECODERS, ENCODERS, SystematicDecoders, SystematicEncoders
from src.dataclasses import SystematicTurboEncoderSpec
from src.utils import snr2sigma
from src.code_trainer import Trainer

parser = argparse.ArgumentParser()

parser.add_argument('encoder', type=SystematicEncoders, choices=list(SystematicEncoders))
parser.add_argument('decoder', type=SystematicDecoders, choices=list(SystematicDecoders))

parser.add_argument('--num_iter', type=int, default=6)
parser.add_argument('--block_len', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--learning_rate', type=float, default=1e-3)

parser.add_argument('--steps_per_epoch', type=int, default=100)
parser.add_argument('--epochs', type=int, default=1024)
parser.add_argument('--snr_train', type=float, default=0.0)


parser.add_argument('--validation_steps', type=int, default=50)

parser.add_argument('--write_logfile', action='store_true')
parser.add_argument('--write_tensorboard', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    encoder = cast(SystematicTurboEncoderSpec, ENCODERS[args.encoder]())
    decoder_factory = DECODERS[args.decoder]

    encoder.systematic_code.normalize_output_table = True
    encoder.interleaved_code.normalize_output_table = True

    sigma = snr2sigma(args.snr_train)
    train_encoder_decoder = TurboSystematicEncoderDecoder(
        systematic_code=encoder.systematic_code,
        interleaved_code=encoder.interleaved_code,
        # channel=AWGN(sigma, power_constraint=scale_constraint),
        channel=AWGN(sigma, power_constraint=identity_constraint),
        decoder_factory=decoder_factory,
        block_len=args.block_len,
        use_max=False,
        num_iter=args.num_iter,
        interleaver=None,
        name='_'.join([str(args.encoder), str(args.decoder)])
    )

    assert train_encoder_decoder.rate == (1, 3)

    optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate, momentum=0.0, nesterov=False)
    trainer = Trainer(
        encoder_decoder=train_encoder_decoder,
        loss=cross_entropy_with_logits,
        optimizer=optimizer,
        block_len=args.block_len,
        batch_size=args.batch_size,
        write_results_to_log=args.write_logfile,
        write_to_tensorboard=args.write_tensorboard
    )

    trainer.train(num_epochs=args.epochs, steps_per_epoch=args.steps_per_epoch, validation_steps=args.validation_steps)

    # encoder_decoders = [
    #     TurboSystematicEncoderDecoder(
    #         encoder.systematic_code, 
    #         encoder.interleaved_code, 
    #         AWGN(sigma), 
    #         # AdditiveTonAWGN(sigma),
    #         # NonIIDMarkovianGaussianAsAWGN(sigma, block_len=args.block_len, p_gb=1.0, p_bg=0.0),
    #         decoder_factory, 
    #         args.block_len, 
    #         False, 
    #         args.num_iter
    #     ) for sigma in test_sigmas]
    
    # assert all(ed.rate == (1, 3) for ed in encoder_decoders)

    # benchmarker = CodeBenchmarker(model_title='_'.join([str(args.encoder), str(args.decoder)]), save_to_file=args.save_to_file)
    # benchmarker.benchmark(encoder_decoders, args.block_len, args.num_blocks, args.batch_size)