# python ./turbo_systematic_benchmark.py turbo-lte hazzys --num_blocks 1000000 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --save_to_file --batch_size 10000
# python ./turbo_systematic_benchmark.py turbo-lte basic --num_blocks 1000000 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --save_to_file --batch_size 10000
# python ./turbo_systematic_benchmark.py turbo-155-7 basic --num_blocks 1000000 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --save_to_file --batch_size 10000
# python ./turbo_systematic_benchmark.py turboae-binary-exact-rsc basic --num_blocks 1000000 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --save_to_file --batch_size 10000

# python ./turbo_systematic_benchmark.py turbo-lte hazzys --num_blocks 1000000 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --save_to_file --batch_size 10000
# python ./turbo_systematic_benchmark.py turbo-155-7 hazzys --num_blocks 1000000 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --save_to_file --batch_size 10000
# python turbo_systematic_trainer.py turboae-binary-exact-rsc hazzys --learning_rate 1e-4 --steps_per_epoch 5600 --epochs 1 --validation_steps 4 --num_iter 6 --batch_size 512 --block_len 100 --write_logfile --write_tensorboard
# python turbo_systematic_trainer.py turbo-lte hazzys --learning_rate 1e-4 --steps_per_epoch 5600 --epochs 1 --validation_steps 4 --num_iter 6 --batch_size 512 --block_len 100 --write_logfile --write_tensorboard
# python turbo_systematic_trainer.py turbo-155-7 hazzys --learning_rate 1e-4 --steps_per_epoch 5600 --epochs 1 --validation_steps 4 --num_iter 6 --batch_size 512 --block_len 100 --write_logfile --write_tensorboard

# python ./turbo_systematic_benchmark.py turboae-approximated-rsc hazzys --num_batches 100 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel awgn
# python ./turbo_nonsystematic_benchmark.py turboae-approximated-nonsys basic --num_batches 100 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel awgn
# python ./turbo_systematic_benchmark.py turboae-approximated-rsc hazzys --num_batches 100 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel atn
# python ./turbo_systematic_benchmark.py turboae-approximated-rsc hazzys --num_batches 10 --num_iter 20 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel awgn
# python ./turbo_nonsystematic_benchmark.py turboae-approximated-nonsys basic --num_batches 100 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel atn
# python ./turbo_systematic_benchmark.py turboae-approximated-rsc hazzys --num_batches 100 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel markov-awgn
# python ./turbo_nonsystematic_benchmark.py turboae-approximated-nonsys basic --num_batches 100 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel markov-awgn

# python ./turbo_nonsystematic_benchmark.py turboae-approximated-nonsys basic --num_batches 10 --num_iter 20 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel awgn
# python ./turbo_systematic_benchmark.py turboae-binary-exact-rsc hazzys --num_batches 10 --num_iter 20 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel awgn
# python ./turbo_nonsystematic_benchmark.py turboae-binary-exact basic --num_batches 10 --num_iter 20 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel awgn
# python ./turbo_systematic_benchmark.py turbo-lte hazzys --num_batches 10 --num_iter 20 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel awgn
# python ./turbo_systematic_benchmark.py turbo-155-7 hazzys --num_batches 10 --num_iter 20 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel awgn

# python ./turbo_systematic_benchmark.py turbo-random5-1-rsc hazzys --num_batches 10 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel awgn
# python ./turbo_systematic_benchmark.py turbo-155-7 hazzys --num_batches 10 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel awgn
# python ./turbo_nonsystematic_benchmark.py turbo-random5-1-nonsys basic --num_batches 10 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel awgn
# python ./turbo_systematic_benchmark.py turbo-random5-2-rsc hazzys --num_batches 10 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel awgn
# python ./turbo_nonsystematic_benchmark.py turbo-random5-2-nonsys basic --num_batches 10 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel awgn
# python ./turbo_systematic_benchmark.py turbo-random5-2-rsc hazzys --num_batches 10 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel awgn
# python ./turbo_systematic_benchmark.py turbo-random5-3-rsc hazzys --num_batches 10 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel awgn
# python ./turbo_nonsystematic_benchmark.py turbo-random5-3-nonsys basic --num_batches 10 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel awgn
# python ./turbo_systematic_benchmark.py turboae-random5-4-rsc hazzys --num_batches 10 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel awgn
# python ./turbo_nonsystematic_benchmark.py turboae-random5-4-nonsys basic --num_batches 10 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel awgn
# python ./turbo_systematic_benchmark.py turboae-random5-5-rsc hazzys --num_batches 10 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel awgn
# python ./turbo_nonsystematic_benchmark.py turboae-random5-5-nonsys basic --num_batches 10 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel awgn

# python ./turbo_systematic_benchmark.py turboae-random5-1-rsc hazzys --num_batches 10 --num_iter 20 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel awgn
# python ./turbo_nonsystematic_benchmark.py turboae-random5-1-nonsys basic --num_batches 10 --num_iter 20 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel awgn
# python ./turbo_systematic_benchmark.py turboae-random5-2-rsc hazzys --num_batches 10 --num_iter 20 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel awgn
# python ./turbo_nonsystematic_benchmark.py turboae-random5-2-nonsys basic --num_batches 10 --num_iter 20 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel awgn
# python ./turbo_systematic_benchmark.py turboae-random5-3-rsc hazzys --num_batches 10 --num_iter 20 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel awgn
# python ./turbo_nonsystematic_benchmark.py turboae-random5-3-nonsys basic --num_batches 10 --num_iter 20 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel awgn

# python ./turbo_systematic_benchmark.py turboae-binary-exact-rsc hazzys --num_batches 10 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel awgn
# python ./turbo_nonsystematic_benchmark.py turboae-binary-exact basic --num_batches 10 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel awgn
# python ./turbo_systematic_benchmark.py turboae-approximated-rsc2 hazzys --num_batches 10 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel awgn
# python ./turbo_systematic_benchmark.py turboae-approximated-rsc2 hazzys --num_batches 10 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel atn
# python ./turbo_systematic_benchmark.py turboae-approximated-rsc2 hazzys --num_batches 10 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel markov-awgn

# python ./turbo_nonsystematic_benchmark.py turboae-binary-exact basic --num_batches 2 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel atn
# python ./turbo_nonsystematic_benchmark.py turboae-binary-exact basic --num_batches 2 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel markov-awgn

# python ./turbo_nonsystematic_benchmark.py turboae-binary-exact basic --num_batches 10 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel atn
# python ./turbo_nonsystematic_benchmark.py turboae-binary-exact basic --num_batches 10 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel markov-awgn
# python ./turbo_systematic_benchmark.py turboae-approximated-rsc2 hazzys --num_batches 10 --num_iter 6 --snr_test_start 3 --snr_test_end 3 --snr_points 1 --write_logfile --write_tensorboard --batch_size 10000 --channel atn

python ./turbo_systematic_benchmark.py turboae-approximated-rsc2 hazzys --num_batches 90 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel awgn
python ./turbo_systematic_benchmark.py turboae-approximated-rsc2 hazzys --num_batches 90 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --write_logfile --write_tensorboard --batch_size 10000 --channel markov-awgn
