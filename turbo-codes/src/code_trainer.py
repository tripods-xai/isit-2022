from collections import defaultdict, OrderedDict
from dataclasses import asdict
import json
from typing import Dict, List
from tqdm import trange
from pprint import pprint
import os
from datetime import datetime
from pytz import timezone

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer
import tensorflow_probability as tfp

from src.dataclasses import TrainerSettings, ValidatorSettings

from .utils import TurboCodesJSONEncoder, safe_open_dir

from .channelcoding.encoder_decoders import EncoderDecoder
from .channelcoding.types import Loss

# ~~TODO: Write history to logfile as we train~~
# ~~TODO: Write to tensorboard~~
# ~~TODO: Normalize output table during training~~
class Validator:

    running_mean_init = tfp.experimental.stats.RunningMean.from_shape(shape=())
    running_var_init = tfp.experimental.stats.RunningVariance.from_shape(shape=())

    def __init__(
        self, 
        encoder_decoder: EncoderDecoder, 
        block_len=100, 
        batch_size=1024, 
        write_results_to_log=False, 
        logdir='./test_logs/', 
        tzname='America/Chicago', 
        results_writer: 'ResultsWriter'=None,
        write_to_tensorboard=False, 
        tensorboard_dir='./tensorboard/testing/',
        verbose=0,
    ):
        self.encoder_decoder = encoder_decoder
        self.block_len = block_len
        self.batch_size = batch_size

        if results_writer is None:
            results_writer = ResultsWriter(self.encoder_decoder.name, logdir=logdir, tzname=tzname)
        self.results_writer = results_writer

        self.logdir = self.results_writer.logdir
        self.model_title = self.results_writer.model_title
        self.tz = self.results_writer.tz
        self.timestamp = self.results_writer.timestamp
        self.model_id = self.results_writer.model_id

        self.write_to_tensorboard = write_to_tensorboard
        self.tensorboard_dir = tensorboard_dir
        self.summary_writer = tf.summary.create_file_writer(os.path.join(safe_open_dir(self.tensorboard_dir), self.model_id)) if self.write_to_tensorboard else None

        self.write_results_to_log = write_results_to_log
        self.verbose = verbose
    
    @tf.function
    def validate_step(self, x_batch):
        _ = self.encoder_decoder(x_batch)
        # print({name: tf.reduce_mean(val) for name, val in self.encoder_decoder.metric_values.items()})
        metric_values = self.encoder_decoder.metric_values
        return metric_values
    
    def validation(self, num_steps, tb_step=0, tb_max_size=100000):
        metric_vars = defaultdict(lambda: Validator.running_var_init)
        self.encoder_decoder.validating()

        tensorboard_value_arrs = {}
        tb_value_arr_max_size = min(tb_max_size, num_steps * self.batch_size)
        iterator = trange(num_steps) if self.verbose > 0 else range(num_steps)
        for i in iterator:
            original_msg = tf.cast(tf.random.uniform((self.batch_size, self.block_len, 1), minval=0, maxval=2, dtype=tf.int32), dtype=tf.float32)
            metric_values = self.validate_step(original_msg)
            
            metric_vars = {name: metric_vars[name].update(tf.reshape(value, (-1,)), axis=0) for name, value in metric_values.items()}
            if self.write_to_tensorboard:
                start_ind = i * self.batch_size
                end_ind = start_ind + self.batch_size
                remaining = tb_value_arr_max_size - start_ind

                for name, val in metric_values.items():
                    if name not in tensorboard_value_arrs:
                        tensorboard_value_arrs[name] = np.empty((tb_value_arr_max_size,) + val.shape[1:], dtype=np.float32)
                    if remaining >= self.batch_size:
                        tensorboard_value_arrs[name][start_ind:end_ind] = val
                    elif 0 < remaining < self.batch_size:
                        tensorboard_value_arrs[name][start_ind:tb_value_arr_max_size] = val[:remaining]

            # self.encoder_decoder.reset()
        
        scores = OrderedDict(sorted(
            [('_'.join([name, 'mean']), value.mean.numpy()) for name, value in metric_vars.items()] + \
            [('_'.join([name, 'var']),value.variance(ddof=1).numpy()) for name, value in metric_vars.items()]
        ))
        
        print('Validation Results:')
        pprint(scores)

        if self.write_to_tensorboard:
            with self.summary_writer.as_default():
                for name, score in scores.items():
                    tf.summary.scalar(name + '_val', score, step=tb_step)
                for name, val in tensorboard_value_arrs.items():
                    tf.summary.histogram(name + '_val', val.ravel(), step=tb_step)

        return scores
    
    def settings(self):
        return ValidatorSettings(
            model_id=self.model_id,
            block_len=self.block_len,
            batch_size=self.batch_size,
            write_results_to_log=self.write_results_to_log,
            logdir=self.logdir,
            tzname=self.tz.zone,
            write_to_tensorboard=self.write_to_tensorboard,
            tensorboard_dir=self.tensorboard_dir,
            verbose=self.verbose
        )
    
    def run(self, steps, tb_step=0):
        results = {"settings": asdict(self.settings()), "encoder_decoder": asdict(self.encoder_decoder.settings()), "steps": steps}
        results["scores"] = self.validation(steps, tb_step=tb_step)

        if self.write_results_to_log:
            self.results_writer.add_results(results)
            self.results_writer.write()

class ResultsWriter:

    def __init__(
        self, 
        model_title: str,
        logdir: str, 
        tzname='America/Chicago'
    ) -> None:
        self.model_title = model_title
        self.tz = timezone(tzname)
        self.timestamp = datetime.now().astimezone(self.tz).strftime('%Y-%m-%d.%H-%M-%S')
        self.model_id = '_'.join([self.model_title, self.timestamp])

        self.logdir = logdir
        self.logfile = self.model_id + '.json'

        self.results_list: List[Dict] = []
    
    def add_results(self, results):
        self.results_list.append(results)

    def write(self):
        with open(os.path.join(safe_open_dir(self.logdir), self.logfile), 'w') as f:
            json.dump(self.results_list, f, cls=TurboCodesJSONEncoder)



class Trainer(Validator):

    def __init__(self, encoder_decoder: EncoderDecoder, loss: str, optimizer: Optimizer, block_len=100, batch_size=1024, write_results_to_log=False, logdir='./training_logs/', tzname='America/Chicago', write_to_tensorboard=False, tensorboard_dir='./tensorboard/training'):
        super().__init__(
            encoder_decoder, 
            block_len=block_len, 
            batch_size=batch_size, 
            write_results_to_log=write_results_to_log,
            logdir=logdir, 
            tzname=tzname, 
            write_to_tensorboard=write_to_tensorboard, 
            tensorboard_dir=tensorboard_dir,
            verbose=0
        )
        self.loss = loss
        self.optmimizer = optimizer


        self.model_title = self.encoder_decoder.name
        self.tz = timezone(tzname)
        self.timestamp = datetime.now().astimezone(self.tz).strftime('%Y-%m-%d.%H-%M-%S')
        self.model_id = '_'.join([self.model_title, self.timestamp])

        self.write_to_tensorboard = write_to_tensorboard
        self.tensorboard_dir = tensorboard_dir
        self.summary_writer = tf.summary.create_file_writer(os.path.join(safe_open_dir(self.tensorboard_dir), self.model_id)) if self.write_to_tensorboard else None

        self.write_results_to_log = write_results_to_log
        self.logdir = logdir
        self.logfile = self.model_id + '.json'
    
    @tf.function
    def train_step(self, x_batch) -> Dict[str, tf.Tensor]:
        with tf.GradientTape(persistent=True) as tape: # type: ignore
            _ = self.encoder_decoder(x_batch)
            metric_values = self.encoder_decoder.metric_values
            loss_value = metric_values[self.loss]
            # print('watched variables')
            # print(tape.watched_variables())
            loss_mean = tf.reduce_mean(loss_value)
            # encoded_mean = tf.reduce_mean(tape.watched_variables()[-1])

        metric_values = self.encoder_decoder.metric_values
        # grads = tape.gradient(tf.reduce_mean(loss_value), self.encoder_decoder.parameters())
        # print('Encoder Decoder Params')
        # print(self.encoder_decoder.parameters())
        # print(tape.gradient(tf.reduce_mean(loss_value), tape.watched_variables()))
        # print('dLoss / dWatched')
        # print(tape.gradient(loss_mean, tape.watched_variables()))
        # for var in tape.watched_variables():
        #     print(f'd{var.name}/dWatched')
        #     print(tape.gradient(var, tape.watched_variables()))
        
        # grads = None
        # print(grads)
        grads = tape.gradient(loss_mean, self.encoder_decoder.parameters())
        self.optmimizer.apply_gradients(zip(grads, self.encoder_decoder.parameters()))
        return metric_values
    
    def train_epoch(self, num_steps, start_tb_step=0):
        metric_means = defaultdict(lambda: self.running_mean_init)
        metric_batch_vars = defaultdict(lambda: self.running_mean_init)
        self.encoder_decoder.training()
        scores = {}

        pbar = trange(num_steps)
        for step in pbar:
            original_msg = tf.cast(tf.random.uniform((self.batch_size, self.block_len, 1), minval=0, maxval=2, dtype=tf.int32), dtype=tf.float32)
            metric_values = self.train_step(original_msg)
            
            for name, value in metric_values.items():
                flat_value = tf.reshape(value, (-1,))
                metric_means[name] = metric_means[name].update(flat_value, axis=0)
                batch_sample_var = self.running_var_init.update(flat_value, axis=0).variance(ddof=1)
                metric_batch_vars[name] = metric_batch_vars[name].update(batch_sample_var)

                if self.write_to_tensorboard:
                    tb_step = start_tb_step + step
                    with self.summary_writer.as_default():
                        mean = tf.reduce_mean(flat_value)
                        tf.summary.scalar(name + '_mean_train', mean, step=tb_step)
                        stddev = tf.sqrt(tf.math.reduce_variance(flat_value) * flat_value.shape[0] / (flat_value.shape[0] - 1))
                        tf.summary.scalar(name + '_std_train', stddev, step=tb_step)
                        tf.summary.histogram(name, flat_value, step=tb_step)



            # self.encoder_decoder.reset()
            
            # We keep track of average batch sample variance since population variance may be different for different batches
            scores = OrderedDict(sorted(
                [('_'.join([name, 'mean']), value.mean.numpy()) for name, value in metric_means.items()] + \
                [('_'.join([name, 'var']), value.mean.numpy()) for name, value in metric_batch_vars.items()],
            ))
            pbar.set_postfix(ordered_dict=scores)
        
        return scores
    
    def run(self, num_epochs, steps_per_epoch, validation_steps=0):
        results = {"settings": asdict(self.settings()), "initial_encoder_decoder": asdict(self.encoder_decoder.settings())}
        results['history'] = [{"epoch": i} for i in range(num_epochs)]
        history = results['history']

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs} for model {self.model_id}")
            train_scores = self.train_epoch(steps_per_epoch)
            history[epoch]['training'] = train_scores

            if validation_steps > 0:
                validation_scores = self.validation(validation_steps)
                history[epoch]['validation'] = validation_scores
            
            self.encoder_decoder.reset()
            history[epoch]['model'] = asdict(self.encoder_decoder.settings())

            if self.write_results_to_log:
                with open(os.path.join(safe_open_dir(self.logdir), self.logfile), 'w') as f:
                    json.dump(results, f, cls=TurboCodesJSONEncoder)
        
        return results
    
    def settings(self):
        return TrainerSettings(
            model_id=self.model_id,
            loss=self.loss,
            optimizer=self.optmimizer.get_config(),
            block_len=self.block_len,
            batch_size=self.batch_size,
            write_results_to_log=self.write_results_to_log,
            logdir=self.logdir,
            tzname=self.tz.zone,
            write_to_tensorboard=self.write_to_tensorboard,
            tensorboard_dir=self.tensorboard_dir
        )
        