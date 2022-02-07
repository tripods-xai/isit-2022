from pprint import pprint
from datetime import datetime
import time
import os
import json
import math

from tqdm.contrib.concurrent import process_map
from tqdm import trange, tqdm

import numpy as np
from numpy.random import default_rng
from commpy import channelcoding as cc

import plotly.express as px

import tests.modified_turbo as mt
from .channels import AWGN, TFAWGN
from .utils import safe_open_dir, sigma2snr, new_seed, NpEncoder, hamming_dist
import .tensorflow_turbo as tt
import .tf_convcode as tcc

ENCODERS = {
    "modified_basic": mt.turbo_encode
}

DECODERS = {
    "modified_basic": mt.turbo_decode,
    "modified_hazzys": mt.hazzys_turbo_decode,
    "tensorflow_basic": tt.turbo_decode_adapter
}

class TurboBenchmarker(object):

    def __init__(self, trellis1, trellis2, systematic=False, model_title: str = "", save_to_file=False, save_plot=False, show_plot=False, seed_generator=default_rng()):
        self.trellis1 = trellis1
        self.trellis2 = trellis2
        self.systematic = systematic
        self.seed_generator = seed_generator
        self.model_title = model_title
        self.save_to_file = save_to_file
        self.save_plot = save_plot
        self.show_plot = show_plot
        self.timestamp = datetime.now().strftime('%Y-%m-%d.%H-%M-%S')
        
        if self.model_title:
            self.model_id = ".".join([model_title, self.timestamp])
        else:
            self.model_id = self.timestamp
        
        self.save_filedir = "./results"
        self.save_filename = ".".join([self.model_id, "json"])
        self.save_plotdir = "./images"
        self.save_ber_plotname = ".".join([self.model_id, "ber", "png"])
        self.save_bler_plotname = ".".join([self.model_id, "bler", "png"])
    
    def batched_turbo_compute(self, encoder_name: str, decoder_name: str, interleaver, sigma: float, block_len: int, num_decoder_iterations: int, batch_size: int, rng):
        channel = AWGN(sigma, rng)
        encoder = ENCODERS[encoder_name]
        decoder = DECODERS[decoder_name]

        message_bits = rng.integers(0, 2, (batch_size, block_len))
        s1, s2, s1_i, s3 = [np.empty(message_bits.shape) for _ in range(4)]
        for i in range(batch_size):
            s1[i], s2[i], s1_i[i], s3[i] = encoder(message_bits[i], self.trellis1, self.trellis2, interleaver)

        s1_r  = channel.corrupt(s1)
        s2_r = channel.corrupt(s2)
        if self.systematic == False:
            s1_i_r = channel.corrupt(s1_i)
        else:
            s1_i_r = interleaver.interlv(s1_r.T).T
        s3_r = channel.corrupt(s3)

        _, decoded_bits = decoder(
            s1_r, s2_r, s1_i_r, s3_r, 
            self.trellis1,
            self.trellis2, 
            channel.sigma ** 2, 
            num_decoder_iterations, 
            interleaver
        )

        num_bit_errors = hamming_dist(message_bits, decoded_bits, axis=1)

        return num_bit_errors
    
    
    def turbo_compute(self, encoder_name: str, decoder_name: str, interleaver, sigma: float, block_len: int, num_decoder_iterations: int, rng):
        channel = AWGN(sigma, rng)
        encoder = ENCODERS[encoder_name]
        decoder = DECODERS[decoder_name]

        message_bits = rng.integers(0, 2, block_len)
        [s1, s2, s1_i, s3] = encoder(message_bits, self.trellis1, self.trellis2, interleaver)

        s1_r  = channel.corrupt(s1)
        s2_r = channel.corrupt(s2)
        if self.systematic == False:
            s1_i_r = channel.corrupt(s1_i)
        else:
            s1_i_r = interleaver.interlv(s1_r)
        s3_r = channel.corrupt(s3)

        _, decoded_bits = decoder(
            s1_r, s2_r, s1_i_r, s3_r, 
            self.trellis1,
            self.trellis2, 
            channel.sigma ** 2, 
            num_decoder_iterations, 
            interleaver
        )

        num_bit_errors = hamming_dist(message_bits, decoded_bits)

        return num_bit_errors
    
    def _turbo_compute_wrapper(self, args):
        return self.turbo_compute(*args)
    
    def _run(self, encoder_name: str, decoder_name: str, test_sigmas, num_block: int, block_len: int, num_decoder_iterations: int, num_cpu: int, batch_size=1):
        settings = {
            "encoder_name": encoder_name,
            "decoder_name": decoder_name,
            "num_block": num_block,
            "block_len": block_len,
            "num_decoder_iterations": num_decoder_iterations,
            "num_cpu": num_cpu,
            "systematic": self.systematic
        }

        interleaver = cc.RandInterlv(block_len, new_seed(self.seed_generator))
        
        for sigma in test_sigmas:
            metrics = {}
            metrics["sigma"] = sigma
            metrics["snr"] = sigma2snr(sigma)
            metrics["settings"] = settings

            print(f"Running experiment with settings")
            pprint(metrics)

            start_time = time.time()
            if batch_size > 1:
                assert decoder_name == "tensorflow_basic"
                results = []
                for i in trange(math.ceil(num_block / batch_size)):
                    cur_batch_size = min(batch_size, num_block - i * batch_size)
                    results.append(self.batched_turbo_compute(encoder_name, decoder_name, interleaver, sigma, block_len, num_decoder_iterations, cur_batch_size, default_rng(new_seed(self.seed_generator))))
                results = np.concatenate(results)
            else:
                args_list = [(encoder_name, decoder_name, interleaver, sigma, block_len, num_decoder_iterations, default_rng(new_seed(self.seed_generator))) for _ in range(num_block)]
                if num_cpu == 1:
                    results = [self.turbo_compute(*args) for args in tqdm(args_list)]
                else:
                    results = process_map(self._turbo_compute_wrapper, args_list, max_workers=num_cpu, chunksize=max(1, num_block // 100))
                results = np.array(results)
            assert len(results) == num_block

            end_time = time.time()
            metrics["number_block_errors"] = np.count_nonzero(results)
            metrics["number_errors"] = np.sum(results)
            metrics["bit_error_rate"] = metrics["number_errors"] / (block_len * num_block)
            metrics["block_error_rate"] = metrics["number_block_errors"] / num_block
            metrics["runtime"] =  end_time - start_time

            yield metrics
    
    def _process_outputs(self, runner):
        results = []
        for test_num, metrics in enumerate(runner):
            print(f"Ran test {test_num}")
            pprint(metrics)
            results.append(metrics)
        total_running_time = sum([m["runtime"] for m in results])
        print(f"Total Running Time: {total_running_time}")
        
        if self.save_to_file:
            with open(self.safe_get_save_filepath(), 'w') as f:
                json.dump(results, f, cls=NpEncoder)
        
        if self.save_plot or self.show_plot:
            snrs = [m["snr"] for m in results]
            bers = [m["bit_error_rate"] for m in results]
            blers = [m["block_error_rate"] for m in results]
            
            ber_fig = px.line(
                x=snrs, y=bers, 
                title=f'{self.model_title} BER vs. SNR', 
                labels={'x': 'SNR', 'y': 'BER'}, log_y=True
            )
            
            bler_fig = px.line(
                x=snrs, y=blers, 
                title=f'{self.model_title} BLER vs. SNR', 
                labels={'x': 'SNR', 'y': 'BLER'}, log_y=True
            )

            if self.save_plot:
                ber_fig.write_image(self.safe_get_ber_plot_filepath())
                bler_fig.write_image(self.safe_get_bler_plot_filepath())
            
            if self.show_plot:
                ber_fig.show()
                bler_fig.show()

    def safe_get_save_filepath(self):
        return os.path.join(safe_open_dir(self.save_filedir), self.save_filename)
    
    def safe_get_ber_plot_filepath(self):
        return os.path.join(self.save_plotdir, self.save_ber_plotname)

    def safe_get_bler_plot_filepath(self):
        return os.path.join(self.save_plotdir, self.save_bler_plotname)
    
    def benchmark(self, encoder_name: str, decoder_name: str, test_sigmas, num_block: int, block_len: int, num_decoder_iterations: int, num_cpu: int, batch_size: int = 1):
        runner = self._run(encoder_name, decoder_name, test_sigmas, num_block, block_len, num_decoder_iterations, num_cpu, batch_size)
        self._process_outputs(runner)

class TFTurboBenchmarker(TurboBenchmarker):

    def __init__(self, trellis1, trellis2, systematic=False, model_title: str = "", save_to_file=False, save_plot=False, show_plot=False, seed_generator=default_rng()):
        super().__init__(trellis1, trellis2, systematic, model_title, save_to_file, save_plot, show_plot, seed_generator=seed_generator)

        self.tf_trellis1 = tt.commpy_trellis_to_tf(self.trellis1)
        self.tf_trellis2 = tt.commpy_trellis_to_tf(self.trellis2)

    def turbo_compute(self, interleaver, sigma: float, block_len: int, num_decoder_iterations: int, batch_size: int):
        channel = TFAWGN(sigma)

        tf_interleaver = tt.commpy_interleaver_to_tf(interleaver)

        message_bits = tf.random.uniform((batch_size, block_len), minval=0, maxval=2, dtype=tf.int32)
        [s1, s2, s1_i, s3] = tt.tf_turbo_encode(message_bits, self.tf_trellis1, self.tf_trellis2, interleaver)

        s1_r  = channel(s1)
        s2_r = channel(s2)
        if self.systematic == False:
            s1_i_r = channel(s1_i)
        else:
            s1_i_r = interleaver.interlv(s1_r)
        s3_r = channel(s3)

        llr = tt.turbo_decode(
            s1_r, s2_r, s1_i_r, s3_r, 
            self.tf_trellis1,
            self.tf_trellis2, 
            tf.zeros(s1_r.shape)
            sigma, 
            num_decoder_iterations, 
            interleaver,
            use_max=True
        )
        decoded_bits = tf.cast(llr > 0, tf.int32)

        num_bit_errors = tf.math.count_nonzero(tf.not_equal(message_bits, decoded_bits), axis=1)

        return num_bit_errors.numpy()
    
    def _run(self, test_sigmas, num_block: int, block_len: int, num_decoder_iterations: int, batch_size=1000):
        settings = {
            "encoder_name": 'tf_encoder',
            "decoder_name": 'tf_max',
            "num_block": num_block,
            "block_len": block_len,
            "num_decoder_iterations": num_decoder_iterations,
            "num_cpu": 0,
            "systematic": self.systematic,
            "batch_size": batch_size
        }

        interleaver = cc.RandInterlv(block_len, new_seed(self.seed_generator))
        
        for sigma in test_sigmas:
            metrics = {}
            metrics["sigma"] = sigma
            metrics["snr"] = sigma2snr(sigma)
            metrics["settings"] = settings

            print(f"Running experiment with settings")
            pprint(metrics)
            
            results = []
            start_time = time.time()
            for i in trange(math.ceil(num_block / batch_size)):
                cur_batch_size = min(batch_size, num_block - i * batch_size)
                results.append(self.turbo_compute(interleaver, sigma, block_len, num_decoder_iterations, cur_batch_size))

            results = np.concatenate(results)

            assert len(results) == num_block

            end_time = time.time()
            metrics["number_block_errors"] = np.count_nonzero(results)
            metrics["number_errors"] = np.sum(results)
            metrics["bit_error_rate"] = metrics["number_errors"] / (block_len * num_block)
            metrics["block_error_rate"] = metrics["number_block_errors"] / num_block
            metrics["runtime"] =  end_time - start_time
        
            yield metrics
    
    def benchmark(self, test_sigmas, num_block: int, block_len: int, num_decoder_iterations: int, batch_size: int = 1):
        runner = self._run(test_sigmas, num_block, block_len, num_decoder_iterations, batch_size)
        self._process_outputs(runner)