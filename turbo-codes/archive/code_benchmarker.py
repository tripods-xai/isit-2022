from pprint import pprint
from datetime import datetime
import time
import os
import json
import math
import itertools as it
from pytz import timezone
from dataclasses import dataclass, asdict


from tqdm import tqdm
import numpy as np

import tensorflow as tf

from ..src.channelcoding.dataclasses import EncoderDecoderSettings
from ..src.channelcoding.encoder_decoders import EncoderDecoder
from ..src.utils import  TurboCodesJSONEncoder, safe_open_dir

@dataclass
class Stats:
    """Stats kept track of by the benchmarker"""
    ber: float
    bler: float
    total_bit_errors: int
    total_block_errors: int
    num_blocks: int
    runtime: float
    ber_var: float
    bler_var: float


@dataclass
class TestRunnerSettings:
    """Settings object for TurboRunner"""
    encoder_decoder_settings: EncoderDecoderSettings
    block_len: int
    num_blocks: int
    batch_size: int
    gpu: bool

@dataclass
class TestResult:
    """Object to hold details on test run"""
    test_number: int
    model_id: str 
    test_run_settings: TestRunnerSettings
    stats: Stats
    

class TestRunner:

    def __init__(self, encoder_decoder: EncoderDecoder, block_len: int, num_blocks: int, batch_size: int):
        self.encoder_decoder = encoder_decoder
        self.block_len = block_len
        self.num_blocks = num_blocks
        self.batch_size = batch_size
        
        self._batch_starts = range(0, self.num_blocks, self.batch_size)
        self._batch_ends = it.chain(self._batch_starts[1:], [self.num_blocks])
    
    def __len__(self):
        return len(self._batch_starts)
    
    @tf.function
    def _run_iter(self, cur_batch_size):
        original_msg = tf.cast(tf.random.uniform((cur_batch_size, self.block_len, 1), minval=0, maxval=2, dtype=tf.int32), dtype=tf.float32)
        decoded_msg = tf.cast(self.encoder_decoder(original_msg) > 0, tf.float32)

        num_bit_errors = tf.reduce_sum(tf.cast(tf.not_equal(original_msg, decoded_msg), tf.float32), axis=1)[:, 0]

        return num_bit_errors
    
    def __iter__(self):
        for batch_start, batch_end in zip(self._batch_starts, self._batch_ends):
            cur_batch_size = batch_end - batch_start 
            
            yield self._run_iter(cur_batch_size).numpy()
    
    def settings(self):
        return TestRunnerSettings(
            encoder_decoder_settings=self.encoder_decoder.settings(), 
            block_len=self.block_len, 
            num_blocks=self.num_blocks, 
            batch_size=self.batch_size,
            gpu=tf.test.is_gpu_available()
        )

class CodeBenchmarker:

    def __init__(self, model_title: str = "", save_to_file=False, save_plot=False, show_plot=False, tzname='America/Chicago'):
        self.model_title = model_title
        self.tz = timezone(tzname)
        self.timestamp = datetime.now().astimezone(self.tz).strftime('%Y-%m-%d.%H-%M-%S')
        
        if self.model_title:
            self.model_id = ".".join([model_title, self.timestamp])
        else:
            self.model_id = self.timestamp
        
        self.save_filedir = "./results"
        self.save_filename = ".".join([self.model_id, "json"])
        self.save_to_file = save_to_file
        
    def _run_wrapper(self, runner):
        """Use this for callbacks"""
        total_bit_errors = 0
        total_block_errors = 0
        ber = 1.
        bler = 1.
        block_ber = np.zeros((runner.num_blocks,), dtype=np.float32)
        block_bler = np.zeros((runner.num_blocks,), dtype=np.int8)
        num_blocks = 0
        num_bits = 0

        pbar = tqdm(runner)
        start_time = time.time()
        for i, bit_errors in enumerate(pbar, 1):
            # Global trackers
            new_num_blocks = num_blocks + bit_errors.shape[0]
            block_ber[num_blocks:new_num_blocks] = bit_errors / runner.block_len
            block_has_error = (bit_errors > 0)
            block_bler[num_blocks:new_num_blocks] = block_has_error

            num_bits += runner.block_len * bit_errors.shape[0]
            num_blocks = new_num_blocks
            total_bit_errors += np.sum(bit_errors)
            total_block_errors += np.count_nonzero(block_has_error)
            ber = total_bit_errors / num_bits
            bler = total_block_errors / num_blocks
            pbar.set_postfix(ber=ber, bler=bler)
        end_time = time.time()

        ber_var = np.var(block_ber, ddof=1)
        bler_var = np.var(block_bler, ddof=1)

        print(len(block_ber))
        print(len(block_bler))
        print(f"BER VAR: {np.sum((block_ber - ber) ** 2) / (runner.num_blocks - 1)}")
        print(f"BLER VAR: {np.sum((block_bler - bler) ** 2) / (runner.num_blocks - 1)}")
        # print(block_ber.tolist())
        # print(block_bler.tolist())

        return Stats(
            total_bit_errors=total_bit_errors,
            total_block_errors=total_block_errors,
            ber=ber,
            bler=bler,
            num_blocks=num_blocks,
            runtime=end_time-start_time,
            ber_var=ber_var,
            bler_var=bler_var
        )
        
    def benchmark(self, encoder_decoders, block_len, num_blocks, batch_size):
        results = []
        for test_num, encoder_decoder in enumerate(encoder_decoders):
            runner = TestRunner(encoder_decoder, block_len, num_blocks, batch_size)
            
            print(f"Running test {test_num}")
            pprint(asdict(runner.settings()))

            stats = self._run_wrapper(runner)
            print(f"Complete test {test_num} for model {self.model_id}")
            pprint(asdict(stats))

            results.append(TestResult(test_number=test_num, model_id=self.model_id, test_run_settings=runner.settings(), stats=stats))
            
            if self.save_to_file:
                self.save_file(results)
        
        return results
    
    def safe_get_save_filepath(self):
        return os.path.join(safe_open_dir(self.save_filedir), self.save_filename)

    def save_file(self, results):
        with open(self.safe_get_save_filepath(), 'w') as f:
                json.dump(results, f, cls=TurboCodesJSONEncoder)
