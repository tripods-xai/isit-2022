import tensorflow as tf
import math
import numpy as np

from commpy.utilities import dec2bitarray, bitarray2dec

from .utils import binlen, batched

@tf.function
def backward_recursion(gamma_values, next_states, batch_size, K, S, reducer):
    # gamma_values = B x K x |S| x 2 : gamma_values[k, i, t] is the gamma for received k from state i to next_states[i, t]
    # next_states = |S| x 2 : next_states[i,t] is the next state after state i with input t
    # B[k][i] = log p(Y[k+1:K-1] | s[k+1] = i) 
    #         = log( Sum over t[ p(Y[k+2:K-1] | s[k+2] = next_states[i, t]) * p(Y[k+1], s[k+2] = next_states[i, t] | s[k+1] = i) ] )
    #         = logsumexp over t[ B[k+1, next_states[i, t]] + gamma_values[k+1, i, t] ]
    B = tf.TensorArray(tf.float32, size=K, clear_after_read=False)
    B = B.write(K-1, tf.zeros((batch_size, S)))
    for k in tf.range(K-2, -1, -1):
        # B x |S| x 2 + B x |S| x 2 -> B x |S|
        # print(B.read(k+1))
        beta = reducer(gamma_values[:, k+1] + tf.gather(B.read(k+1), next_states, axis=1), 2)
        B = B.write(k, beta - reducer(beta, axis=1, keepdims=True))
    return tf.transpose(B.stack(), perm=[1, 0, 2])

@tf.function
def forward_recursion(previous_gamma_values, previous_states, batch_size, K, S, reducer):
    # gamma_values = B x K x |S| x 2 : gamma_values[k, i, t] is the gamma for received k from state i to next_states[i, t]
    # previous_states = |S| x 2 x 2 : previous_states[j] are the pair of previous state that gets to j and the input to move to j
    # A[k][j] = log p(Y[0:k-1], s[k] = j)
    #         = log( Sum over r[ p(Y[0:k-2], s[k-1]=previous_states[j, r, 0]) * p(Y[k-1], s[k]=j | s[k-1]=previous_states[j, r, 0]) ] )
    #         = logsumexp over r[ A[k-1, previous_states[j, r, 0]] + previous_gamma_values[k-1, j, r] ] ] 
    
    init_row = tf.tile(tf.constant([0.] + [-np.inf]*(S-1))[None, :], [batch_size, 1])
    A = tf.TensorArray(tf.float32, size=K, clear_after_read=False)
    A = A.write(0, init_row)
    for k in tf.range(1, K):
        # B x |S| x 2 + B x |S| x 2 -> B x |S|
        previous_alphas = tf.gather(A.read(k-1), previous_states[:, :, 0], axis=1)
        alpha = reducer(previous_gamma_values[:, k-1] + previous_alphas, 2)
        A = A.write(k, alpha - reducer(alpha, axis=1, keepdims=True))
    return tf.transpose(A.stack(), perm=[1, 0, 2])

@tf.function
def map_decode(received_symbols, code_outputs, next_states, previous_states, L_int, noise_std, use_max=False):
    if use_max:
        reducer = tf.math.reduce_max
    else:
        reducer = tf.math.reduce_logsumexp
    
    # received_symbols = B x K x n : 1 / n is code rate
    # code_outputs = |S| x 2 x n : code_outputs[i,t] is the codeword emitted at state i with input t
    # next_states = |S| x 2 : next_states[i,t] is the next state after state i with input t
    # previous_states = |S| x 2 x 2 : previous_states[j] are the pair of previous state that gets to j and the input to move to j
    # L_int = B x K : the prior LLR for x_k = 1
    # noise_variance : the variance of the AWGN
    batch_size = received_symbols.shape[0]
    K = received_symbols.shape[1]
    n = received_symbols.shape[2]
    S = code_outputs.shape[0]
    noise_variance = tf.square(noise_std)
    
    # Compute ln(Chi) values
    # chi_values[k, i, t] = log p(Y[k] | s[k] = i, s[k+1] = next_states[i, t])
    # B x K x 1 x 1 x n - 1 x 1 x |S| x 2 x n, reduce on 4th axis, result is B x K x |S| x 2
    square_noise_sum = tf.math.reduce_sum(tf.square(received_symbols[:, :, None, None, :] - code_outputs[None, None, :, :, :]), axis=4)
    # chi_values = -tf.math.log(noise_std * tf.math.sqrt(2 * math.pi)) - 1 / (2 * noise_variance) * square_noise_sum
    # the first log term will cancel out in calculation of LLRs so I can drop it
    chi_values = - 1. / (2 * noise_variance) * square_noise_sum
        
    # Compute ln(Gamma) values
    # gamma_values[k, i, t] = log p(Y[k], s[k+1] = next_states[i, t] | s[k] = i) = log p(s[k+1] = next_states[i, t] | s[k] = i) + chi_values[k, i, t]
    # B x K x 2
    transition_prob_values = tf.stack([tf.math.log_sigmoid(-L_int), tf.math.log_sigmoid(L_int)], axis=2)
    # B x K x |S| x 2
    gamma_values = chi_values + transition_prob_values[:, :, None, :]
    
    # Compute ln(B)
    # B x K x |S|
    B = backward_recursion(gamma_values, next_states, batch_size, K, S, reducer)
    
    # Compute ln(A)
    previous_gamma_values = tf.transpose(
        tf.gather_nd(tf.transpose(gamma_values, perm=[2, 3, 0, 1]), previous_states), 
        perm=[2, 3, 0, 1]
    )
    # B x K x |S|
    A = forward_recursion(previous_gamma_values, previous_states, batch_size, K, S, reducer)

    # Compute L_ext
    # L = log Sum over i[ p(Y[0:K-1], s_k=i, s_k+1=next_states[i, 1]) ] / "
    # = log Sum over i[ p(Y[0:k-1], Y[k], Y[k+1:K-1], s_k=i, s_k+1=next_states[i, 1]) ] / "
    # = log Sum over i[ p(Y[0:k-1], s_k=i) * p(Y[k], s_k+1=next_states[i, 1] | s_k=i) * P(Y[k+1:K-1] | s_k+1=next_states[i, 1]) ] / "
    # = log Sum over i[ p(Y[0:k-1], s_k=i) * p(Y[k] | s_k+1=next_states[i, 1], s_k=i) * p(s_k+1=next_states[i, 1] | s_k=i) * p(Y[k+1:K-1] | s_k+1=next_states[i, 1]) ] / "
    # = log Sum over i[ p(Y[0:k-1], s_k=i) * p(Y[k] | s_k+1=next_states[i, 1], s_k=i) * p(x_k=1) * p(Y[k+1:K-1] | s_k+1=next_states[i, 1]) ] / "
    # = log( p(x_k=1) / p(x_k=0) ) * log Sum over i[ p(Y[0:k-1], s_k=i) * p(Y[k] | s_k+1=next_states[i, 1], s_k=i) * p(Y[k+1:K-1] | s_k+1=next_states[i, 1]) ] / "
    # = L_int + logsumexp over i[ A[k, i] + chi_values[k, i, 1] + B[k, next_states[i, 1]] ] - logsumexp over i[ A[k, i] + chi_values[k, i, 0] + B[k, next_states[i, 1]] ] 
    # = L_int + L_ext
    # -> L_ext = logsumexp over i[ A[k, i] + chi_values[k, i, 1] + B[k, next_states[i, 1]] ] - logsumexp over i[ A[k, i] + chi_values[k, i, 0] + B[k, next_states[i, 1]] ]
    
    B_next_states = tf.gather(B, next_states, axis=2)
    L_ext = reducer(A + chi_values[:, :, :, 1] + B_next_states[:, :, :, 1], axis=2) - reducer(A + chi_values[:, :, :, 0] + B_next_states[:, :, :, 0], axis=2)

    return L_ext

@tf.function
def turbo_decode(stream1, stream2, stream1_i, stream3, tf_trellis1, tf_trellis2, tf_interleaver, L_int, noise_std, num_iter=6, use_max=False):
    L_int1 = L_int
    L_ext1 = tf.zeros_like(L_int1)
    received1 = tf.stack([stream1, stream2], axis=2)
    
    # stream1_i = tf.gather(stream1, tf_interleaver["permutation"])
    received2 = tf.stack([stream1_i, stream3], axis=2)
    for i in tf.range(num_iter):
        prev_l_int1 = L_int1
        L_ext1 = map_decode(received1, tf_trellis1["code_outputs"], tf_trellis1["next_states"], tf_trellis1["previous_states"], L_int1, noise_std, use_max)

        L_int2 = tf.gather(L_ext1, tf_interleaver["permutation"], axis=1)
        L_ext2 = map_decode(received2, tf_trellis2["code_outputs"], tf_trellis2["next_states"], tf_trellis2["previous_states"], L_int2, noise_std, use_max)

        L_int1 = tf.gather(L_ext2, tf_interleaver["depermutation"], axis=1)

    L = L_ext1 + L_int1
    return L

# def decode_network(outputs)

def commpy_trellis_to_tf(trellis):
    tf_trellis = {}
    tf_trellis["next_states"] = tf.constant(trellis.next_state_table, dtype=tf.int32)
    
    num_states = trellis.output_table.shape[0]
    num_inputs = trellis.output_table.shape[1]
    n = np.amax(np.vectorize(binlen)(trellis.output_table))

    code_outputs = np.zeros((num_states, num_inputs, n))
    for state in range(num_states):
        for input_sym in range(num_inputs):
            code_outputs[state,input_sym] = dec2bitarray(trellis.output_table[state,input_sym], n)
    tf_trellis["code_outputs"] = 2. * tf.constant(code_outputs, dtype=tf.float32) - 1.
    
    previous_states = [[] for _ in range(num_states)]
    for state in range(num_states):
        for input_sym in range(num_inputs):
            next_state = trellis.next_state_table[state, input_sym]
            previous_states[next_state].append([state, input_sym])
    tf_trellis["previous_states"] = tf.constant(previous_states, dtype=tf.int32)

    return tf_trellis

def commpy_interleaver_to_tf(interleaver):
    tf_interleaver = {}
    tf_interleaver["permutation"] = tf.constant(interleaver.p_array, dtype=tf.int32)
    s_array = np.zeros_like(interleaver.p_array)
    s_array[interleaver.p_array] = np.arange(len(interleaver.p_array))
    tf_interleaver["depermutation"] = tf.constant(s_array, dtype=tf.int32)
    return tf_interleaver

def turbo_decode_adapter(s1_r, s2_r, s1_i_r, s3_r, trellis1, trellis2, noise_variance, num_decoder_iterations, interleaver):
    tf_trellis1 = commpy_trellis_to_tf(trellis1)
    tf_trellis2 = commpy_trellis_to_tf(trellis2)
    tf_interleaver = commpy_interleaver_to_tf(interleaver)

    stream1 = tf.constant(batched(s1_r, 1), dtype=tf.float32)
    stream2 = tf.constant(batched(s2_r, 1), dtype=tf.float32)
    stream1_i = tf.constant(batched(s1_i_r, 1), dtype=tf.float32)
    stream3 = tf.constant(batched(s3_r, 1), dtype=tf.float32)

    L = turbo_decode(stream1, stream2, stream1_i, stream3, tf_trellis1, tf_trellis2, tf_interleaver, tf.zeros(stream1.shape), math.sqrt(noise_variance), num_iter=num_decoder_iterations)
    return L.numpy().reshape(s1_r.shape), tf.cast(L > 0, tf.int32).numpy().reshape(s1_r.shape)

def tf_conv_encode(msg_bits, tf_trellis):
    """
    tf_trellis:
        - "code_outputs"
        - "next_states"
        - "previous_states"
    """
    batch_size = msg_bits.shape[0]
    num_inbits = msg_bits.shape[1]
    state = 0
    outbits_batch = tf.TensorArray(size=num_inbits)
    for i in tf.range(num_inbits):
        inbit_batch = msg_bits[:, i]
        # Makes the assumption that k=1
        outbits_batch = outbits_batch.write(i, tf.gather(tf_trellis["code_outputs"][state], inbit_batch, axis=0))
        # print(f"[MCC] Time {i} current state {state}")
        # print(f"[MCC] Time {i} output val {out_dec}")

        state = trellis["next_states"][state, inbit]

    return tf.transpose(outbits_batch.stack(), perm=[1, 0])

def tf_turbo_encode(msg_btis, tf_trellis1, tf_trellis2, tf_interleaver):
    stream = tf_conv_encode(msg_bits, tf_trellis1)
    stream1 = stream[:, ::2]
    stream2 = stream[:, 1::2]

    interlv_msg_bits = tf.gather(msg_bits, tf_interleaver["permutation"], axis=1)
    interlv_stream = tf_conv_encode(interlv_msg_bits, tf_trellis2)
    stream1_i = interlv_stream[:, ::2]
    stream3 = interlv_stream[:, 1::2]

    return [stream1, stream2, stream1_i, stream3]