# python ./modified_turbo_codes_benchmark.py --num_block 1000 --block_len 100 --num_dec_iteration 100 --snr_test_start -1.5 --snr_test_end 5.0 --snr_points 14 --num_cpu 12 --save --plot
# python ./turbo_codes_benchmark.py --num_block 1000 --block_len 100 --num_dec_iteration 100 --snr_test_start -1.5 --snr_test_end 4.0 --snr_points 12 --num_cpu 12 --save --plot

# python ./modified_turbo_codes_benchmark.py --num_block 1000 --block_len 100 --num_dec_iteration 100 --g1 31 --g2 23 --g3 30 --b1 1 --b2 0 --b3 1 --snr_test_start -3.0 --snr_test_end 3.0 --snr_points 13 --num_cpu 12 --save --plot
# python ./turbo_codes_benchmark.py --num_block 100 --block_len 100 --num_dec_iteration 100 --g1 31 --g2 23 --g3 30 --feedback 0 --snr_test_start -1.0 --snr_test_end 1.0 --snr_points 3 --num_cpu 12 --save --plot

# Test
# python ./modified_turbo_codes_benchmark.py --num_block 1000 --block_len 100 --num_dec_iteration 20 --g1 31 --g2 23 --g3 30 --b1 1 --b2 0 --b3 1 --rsc --snr_test_start -1.0 --snr_test_end 1.0 --snr_points 3 --num_cpu 12
# python ./turbo_codes_benchmark.py --num_block 1000 --block_len 100 --num_dec_iteration 20 --g1 7 --g2 5 --g3 5 --feedback 7 --systematic --snr_test_start -1.0 --snr_test_end 1.0 --snr_points 3 --num_cpu 12

# python ./modified_turbo_codes_benchmark.py --num_block 10000 --block_len 100 --num_dec_iteration 20 --g1 31 --g2 23 --g3 30 --b1 1 --b2 0 --b3 1 --rsc --snr_test_start -3.0 --snr_test_end 1.0 --snr_points 9 --num_cpu 12 --save --plot
# python ./modified_turbo_codes_benchmark.py --num_block 60000 --block_len 100 --num_dec_iteration 20 --g1 31 --g2 23 --g3 30 --b1 1 --b2 0 --b3 1 --rsc --snr_test_start 1.5 --snr_test_end 3.0 --snr_points 4 --num_cpu 12 --save --plot
# python ./turbo_codes_benchmark.py --num_block 10000 --block_len 100 --num_dec_iteration 20 --g1 7 --g2 5 --g3 5 --feedback 7 --systematic --snr_test_start -3.0 --snr_test_end 1.0 --snr_points 9 --num_cpu 12 --save --plot
# python ./turbo_codes_benchmark.py --num_block 60000 --block_len 100 --num_dec_iteration 20 --g1 7 --g2 5 --g3 5 --feedback 7 --systematic --snr_test_start 1.5 --snr_test_end 3.0 --snr_points 4 --num_cpu 12 --save --plot

# python ./modified_turbo_codes_benchmark.py --num_block 10000 --block_len 100 --num_dec_iteration 20 --g1 31 --g2 23 --g3 30 --b1 1 --b2 0 --b3 1 --rsc --snr_test_start -3.0 --snr_test_end 1.0 --snr_points 9 --num_cpu 1 --decoder "tensorflow_basic" --batch_size 500

# python ./turbo_systematic_benchmark.py turbo-155-7 basic --num_blocks 1000000 --num_iter 6 --snr_test_start 2 --snr_test_end 3 --snr_points 3 --save_to_file --batch_size 10000
python ./turbo_systematic_benchmark.py turboae-binary-exact-rsc hazzys --num_blocks 1000000 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --save_to_file --batch_size 10000
# python ./turbo_nonsystematic_benchmark.py turboae-binary-exact basic --num_blocks 1000000 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --save_to_file --batch_size 10000
# python ./turbo_systematic_benchmark.py turboae-approximated-rsc hazzys --num_blocks 1000000 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --save_to_file --batch_size 10000
# python ./turbo_nonsystematic_benchmark.py turboae-approximated-nonsys basic --num_blocks 1000000 --num_iter 6 --snr_test_start -3 --snr_test_end 3 --snr_points 13 --save_to_file --batch_size 10000

# python ./turbo_nonsystematic_benchmark.py turboae-approximated-nonsys basic --num_blocks 1000000 --num_iter 6 --snr_test_start 0.5 --snr_test_end 3 --snr_points 6 --save_to_file --batch_size 10000

# python ./turbo_systematic_benchmark.py turbo-155-7 hazzys --num_blocks 1000000 --num_iter 6 --snr_test_start 1.5 --snr_test_end 3 --snr_points 4 --save_to_file --batch_size 10000
# python ./turbo_systematic_benchmark.py turbo-155-7 hazzys --num_blocks 1000000 --num_iter 6 --snr_test_start 1.5 --snr_test_end 3 --snr_points 4 --save_to_file --batch_size 10000
# python ./turbo_systematic_benchmark.py turbo-155-7 hazzys --num_blocks 1000000 --num_iter 6 --snr_test_start 1.5 --snr_test_end 3 --snr_points 4 --save_to_file --batch_size 10000
# python ./turbo_systematic_benchmark.py turbo-155-7 hazzys --num_blocks 1000000 --num_iter 6 --snr_test_start 1.5 --snr_test_end 3 --snr_points 4 --save_to_file --batch_size 10000

# python ./turbo_systematic_benchmark.py turboae-binary-exact-rsc hazzys --num_blocks 1000000 --num_iter 6 --snr_test_start 1.5 --snr_test_end 3 --snr_points 4 --save_to_file --batch_size 10000
# python ./turbo_systematic_benchmark.py turboae-binary-exact-rsc hazzys --num_blocks 1000000 --num_iter 6 --snr_test_start 1.5 --snr_test_end 3 --snr_points 4 --save_to_file --batch_size 10000
# python ./turbo_systematic_benchmark.py turboae-binary-exact-rsc hazzys --num_blocks 1000000 --num_iter 6 --snr_test_start 1.5 --snr_test_end 3 --snr_points 4 --save_to_file --batch_size 10000
# python ./turbo_systematic_benchmark.py turboae-binary-exact-rsc hazzys --num_blocks 1000000 --num_iter 6 --snr_test_start 1.5 --snr_test_end 3 --snr_points 4 --save_to_file --batch_size 10000

# python ./turbo_nonsystematic_benchmark.py turboae-binary-exact basic --num_blocks 1000000 --num_iter 6 --snr_test_start 1.5 --snr_test_end 3 --snr_points 4 --save_to_file --batch_size 10000
# python ./turbo_nonsystematic_benchmark.py turboae-binary-exact basic --num_blocks 1000000 --num_iter 6 --snr_test_start 1.5 --snr_test_end 3 --snr_points 4 --save_to_file --batch_size 10000
# python ./turbo_nonsystematic_benchmark.py turboae-binary-exact basic --num_blocks 1000000 --num_iter 6 --snr_test_start 1.5 --snr_test_end 3 --snr_points 4 --save_to_file --batch_size 10000
# python ./turbo_nonsystematic_benchmark.py turboae-binary-exact basic --num_blocks 1000000 --num_iter 6 --snr_test_start 1.5 --snr_test_end 3 --snr_points 4 --save_to_file --batch_size 10000