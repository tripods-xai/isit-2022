# Enviromnent
- Gurobi 9.1.2
- Python 3.8.8
- numpy 1.20.1


# Parameter
- -num_smpl: the number of samples running in the optimization problem (up to 100 for the given data set). Default: 100.
- -shift_length: the shift bit length. Default: 4.
- -block_idx: the index of the encoding blocks. There have three blocks. Default: 1.
- -memory_length: the memory length. Default: 10.
- -multiple_solution: 0 for single(optimal) solution, and 1 for mutiple solutions (showing the 5 optimal solutions). Default: 0.

# Run
Example:
We want to generate one optimal result of generator 2 (for the encoding block 2), for given 10 data samples. We consider the shift to be 4 and the memory length to be 10. 


`python main.py -num_smpl 10 -shift_length 4 -block_idx 2 -memory_length 10 -multi_solution 0`

The returnning result should be g = 00101110000 with cost 0.018 (1.8%). 
