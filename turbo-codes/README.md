# turbo-codes
Tensorflow based encoder and decoder for turbo codes.

## Basic Usage Directions

### Setup

#### Running directly on your computer

**I highly recommend you create your own conda or pip environment for this. This project requires python3.**

Once you are in you environment, run
```
pip install -r requirements.txt
```
This will install all the relevant packages.

#### Running in a Docker Container

**Currently the Docker Container is setup to require a GPU. For instructions on removing this requirement, please see below.**

I've included a script to make setup easy. However, keep in mind, a docker container can take a lot of space, so make sure you have several GB free.

Simply run
```
./run_docker.sh
```
This should build the Docker Container.

To remove the GPU requirement, change line 1 in `Dockerfile` from
```
FROM tensorflow/tensorflow:latest-gpu-jupyter
```
to
```
FROM tensorflow/tensorflow:latest-jupyter
```

### Running the benchmark code
First I recommend familiarizing yourself with the help menus for both `turbo_systematic_benchmark.py` and `turbo_nonsystematic_benchmark.py`. I've included them below for convenience:

```
usage: turbo_systematic_benchmark.py [-h] [--num_batches NUM_BATCHES] [--block_len BLOCK_LEN] [--num_iter NUM_ITER] [--batch_size BATCH_SIZE] [--snr_test_start SNR_TEST_START] [--snr_test_end SNR_TEST_END]
                                     [--snr_points SNR_POINTS] [--channel {awgn,atn,markov-awgn}] [--write_logfile] [--write_tensorboard]
                                     {turbo-155-7,turboae-binary-exact-rsc,turboae-approximated-rsc,turbo-12330-31-rsc,turbo-lte,turbo-random-7,turbo-random5-1-rsc,turbo-random5-2-rsc,turbo-random5-3-rsc,turbo-random5-4-rsc,turbo-random5-5-rsc,turboae-approximated-rsc2}
                                     {hazzys,basic}

positional arguments:
  {turbo-155-7,turboae-binary-exact-rsc,turboae-approximated-rsc,turbo-12330-31-rsc,turbo-lte,turbo-random-7,turbo-random5-1-rsc,turbo-random5-2-rsc,turbo-random5-3-rsc,turbo-random5-4-rsc,turbo-random5-5-rsc,turboae-approximated-rsc2}
                        The encoder you wish to test against. See paper for more details.
  {hazzys,basic}        The kind of BCJR decoder to use. 'basic' is a traditional BCJR decoder, and 'hazzys' is a systematic code specific decoder that slightly improves performance.

options:
  -h, --help            show this help message and exit
  --num_batches NUM_BATCHES
                        Number of batches to sample and benchmark code on. `num_batches`*`batch_size` = total number of blocks tested.
  --block_len BLOCK_LEN
                        Block length at which to run the code.
  --num_iter NUM_ITER   Number of iterations to run BCJR decoder.
  --batch_size BATCH_SIZE
                        Batch size for benchmarking. Larger batch uses more memory but can speed up benchmarking.`num_batches`*`batch_size` = total number of blocks tested.
  --snr_test_start SNR_TEST_START
                        The low point for sampling a range of SNRs to test at.
  --snr_test_end SNR_TEST_END
                        The high point for sampling a range of SNRs to test at.
  --snr_points SNR_POINTS
                        Number of SNR testing points to be sampled. They will be evenly distributed along `snr_test_start` and `snr_test_end`.
  --channel {awgn,atn,markov-awgn}
                        Noisy channel to test against. Default is AWGN.
  --write_logfile       This will write your results and many more details to a logfile in the test_logs folder. It will automatically create the folder if there is none. These logfiles will store BER
                        performance along with other helpful metrics.
  --write_tensorboard   This will write the results in the logfile to Tensorboard as well. You can then use Tensorboard to visualize those results.
```

```
usage: turbo_nonsystematic_benchmark.py [-h] [--num_batches NUM_BATCHES] [--block_len BLOCK_LEN] [--num_iter NUM_ITER] [--batch_size BATCH_SIZE] [--snr_test_start SNR_TEST_START] [--snr_test_end SNR_TEST_END]
                                        [--snr_points SNR_POINTS] [--channel {awgn,atn,markov-awgn}] [--write_logfile] [--write_tensorboard]
                                        {turbo-755-0,turboae-binary-exact,turboae-approximated-nonsys,turbo-random5-1-nonsys,turbo-random5-2-nonsys,turbo-random5-3-nonsys,turbo-random5-4-nonsys,turbo-random5-5-nonsys}
                                        {basic}

positional arguments:
  {turbo-755-0,turboae-binary-exact,turboae-approximated-nonsys,turbo-random5-1-nonsys,turbo-random5-2-nonsys,turbo-random5-3-nonsys,turbo-random5-4-nonsys,turbo-random5-5-nonsys}
                        The nonsystematic encoder you wish to test against. The help menu will show the available options. See paper for more details.
  {basic}               The kind of BCJR decoder to use. 'basic' is a traditional BCJR decoder reworked for nonsystematic codes. This is the only type.

options:
  -h, --help            show this help message and exit
  --num_batches NUM_BATCHES
                        Number of batches to sample and benchmark code on. `num_batches`*`batch_size` = total number of blocks tested.
  --block_len BLOCK_LEN
                        Block length at which to run the code.
  --num_iter NUM_ITER   Number of iterations to run BCJR decoder.
  --batch_size BATCH_SIZE
                        Batch size for benchmarking. Larger batch uses more memory but can speed up benchmarking.`num_batches`*`batch_size` = total number of blocks tested.
  --snr_test_start SNR_TEST_START
                        The low point for sampling a range of SNRs to test at.
  --snr_test_end SNR_TEST_END
                        The high point for sampling a range of SNRs to test at.
  --snr_points SNR_POINTS
                        Number of SNR testing points to be sampled. They will be evenly distributed along `snr_test_start` and `snr_test_end`.
  --channel {awgn,atn,markov-awgn}
                        Noisy channel to test against. Default is AWGN.
  --write_logfile       This will write your results and many more details to a logfile in the test_logs folder. The logfile will be named based on the 'model-id' printed to stdout. It can easily be identified
                        as it will have a timestamp in the name. It will automatically create the folder if there is none. These logfiles will store BER performance along with other helpful metrics.
  --write_tensorboard   This will write the results in the logfile to Tensorboard as well. You can then use Tensorboard to visualize those results.
```

You can use these settings to produce data for whatever encoders, channels, decoders, SNRS, and sample sizes you need. Below are some examples that will produce the same data (within error bounds) as in the original paper:

```
python ./turbo_systematic_benchmark.py turbo-155-7 hazzys --block_len 100 --num_batches 100 --batch_size 10000 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --channel awgn --write_logfile
python ./turbo_nonsystematic_benchmark.py turbo-755-0 basic --block_len 100 --num_batches 100 --batch_size 10000 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --channel awgn --write_logfile
```

They will run the above encoder for a total of 1,000,000 test samples.

**Before running these I recommend trying a quicker command to make sure you have any issues sorted out:**
```
python ./turbo_systematic_benchmark.py turbo-155-7 hazzys --block_len 100 --num_batches 2 --batch_size 10000 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 2 --channel awgn --write_logfile
python ./turbo_nonsystematic_benchmark.py turbo-755-0 basic --block_len 100 --num_batches 2 --batch_size 10000 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 2 --channel awgn --write_logfile
```

### Reading the Benchmarking Output

As mentioned in the help menus, you can find the output in a JSON file in the `test_logs` folder. Upon running, the JSON file will be one ugly line, so you should run it through your favorite formatter to make it look nice. To plot it you can use python's `json` library to extract the data and `matplotlib` to plot it. See [this example](test_logs/example.json) JSON file. In the JSON file you will find exhaustive details about all the settings used during the benchmarking experiment, along with scores. 

Scores can be found under the `scores` key in the JSON file. There will be one set of `scores` for each SNR you are running, and look at the `channel` key for the SNR. In [example.json](test_logs/example.json):
```json
[
  {
    "encoder_decoder": {
      ...
      "channel": {"sigma": 1.4125375446227544, "snr": -3, "name": "AWGN"},
      ...
    },
    "steps": 2,
    "scores": {
      "ber_mean": 0.12335102260112762,
      "ber_var": 0.008520855568349361,
      "bler_mean": 0.9909000396728516,
      "bler_var": 0.00901764165610075,
      "cross_entropy_mean": 0.29167234897613525,
      "cross_entropy_var": 0.3835274875164032,
      "encoded_mean": 0.5000291466712952,
      "encoded_var": 0.2500000298023224
    }
  },
  {
    "encoder_decoder": {
      ...
      "channel": {
        "sigma": 0.7079457843841379,
        "snr": 3.0000000000000004,
        "name": "AWGN"
      },
      ...
    },
    "steps": 2,
    "scores": {
      "ber_mean": 0.00017250000382773578,
      "ber_var": 0.000002245356199637172,
      "bler_mean": 0.014999999664723873,
      "bler_var": 0.014775731600821018,
      "cross_entropy_mean": 0.0005221577594056726,
      "cross_entropy_var": 0.0014876214554533362,
      "encoded_mean": 0.49978017807006836,
      "encoded_var": 0.2500000596046448
    }
  }
]
```

If you have difficulty with plotting the results, please reach out to me at abhmul@gmail.com or mulgund2@uic.edu.


