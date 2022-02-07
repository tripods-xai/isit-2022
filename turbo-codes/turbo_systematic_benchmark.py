__author__ = 'abhmul'
'''
Evaluate
'''
from dataclasses import asdict
from pprint import pprint
from typing import cast
from src.channelcoding.encoder_decoders import TurboSystematicEncoderDecoder
from src.channels import CHANNELS, NoisyChannels
from src.code_trainer import ResultsWriter, Validator
from src.codes import DECODERS, ENCODERS, SystematicDecoders, SystematicEncoders
from src.dataclasses import SystematicTurboEncoderSpec
from src.utils import get_test_sigmas

import argparse


parser = argparse.ArgumentParser()

parser.add_argument('encoder', type=SystematicEncoders, choices=list(SystematicEncoders))
parser.add_argument('decoder', type=SystematicDecoders, choices=list(SystematicDecoders))

parser.add_argument('--num_batches', type=int, default=1)
parser.add_argument('--block_len', type=int, default=100)
parser.add_argument('--num_iter', type=int, default=6)
parser.add_argument('--batch_size', type=int, default=1000)

parser.add_argument('--snr_test_start', type=float, default=-1.5)
parser.add_argument('--snr_test_end', type=float, default=2.0)
parser.add_argument('--snr_points', type=int, default=8)

parser.add_argument('--channel', type=NoisyChannels, choices=list(NoisyChannels), default=NoisyChannels.AWGN)

parser.add_argument('--write_logfile', action='store_true')
parser.add_argument('--write_tensorboard', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    snrs, test_sigmas = get_test_sigmas(
        args.snr_test_start, args.snr_test_end, args.snr_points)
    
    encoder = cast(SystematicTurboEncoderSpec, ENCODERS[args.encoder]())
    decoder_factory = DECODERS[args.decoder]
    channel = CHANNELS[args.channel]

    model_title = '_'.join([str(args.encoder), str(args.decoder)])
    encoder_decoders = [
        TurboSystematicEncoderDecoder(
            encoder.systematic_code, 
            encoder.interleaved_code, 
            channel(sigma=sigma, block_len=args.block_len),
            decoder_factory, 
            args.block_len, 
            False, 
            args.num_iter,
            name=model_title
        ) for sigma in test_sigmas]
    
    assert all(ed.rate == (1, 3) for ed in encoder_decoders)

    results_writer = ResultsWriter(
        model_title=model_title, 
        # debugging
        # logdir='./tmp/test_logs/'
        logdir='./test_logs/'
    )

    validators = [
        Validator(
            enc_dec, 
            block_len=args.block_len, 
            batch_size=args.batch_size, 
            write_results_to_log=args.write_logfile,
            results_writer=results_writer,
            write_to_tensorboard=args.write_tensorboard,
            # debugging
            # tensorboard_dir='./tmp/tensorboard_dir/',
            tensorboard_dir='./tensorboard/testing/',
            verbose=1
        )
        for enc_dec in encoder_decoders
    ]

    steps = args.num_batches
    for i, validator in enumerate(validators, start=1):
        print(f'Running test {i}/{len(validators)} for snr {snrs[i-1]}')
        pprint(asdict(validator.settings()))
        validator.run(steps, tb_step=i)