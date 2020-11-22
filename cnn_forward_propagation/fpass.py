"""## Forward Propagation"""

import numpy as np
from math import ceil


def add_zero_padding(X, padding, mode='constant'):
    """Adds padding to images in dataset X."""

    pad_width = ((0,0), (padding,padding), (padding,padding), (0,0))
    X_padded = np.pad(X, pad_width=pad_width, mode=mode, constant_values=0)
    return X_padded


def convolution_operation(a_in, W, b):
    """Computes the convolution operation."""
    
    conv_out = np.sum(a_in * W) + float(b)
    return conv_out


def relu(x):
    z = np.zeros_like(x)
    return np.maximum(x, z)


def channels_adjustment(M):
    """If number of channels is 1 but shape doesn't adjust for its dimension,
    reshapes the matrix to dimension denoting number of channels as 1."""
    
    if len(M.shape) == 3:
        M = np.expand_dims(M, axis=3)
    return M


def CNN_node_forward(A_in, W, b, padding, stride, G=relu):
    """Performs Forward Pass through a single convolution unit."""

    # Adjusting for the number of channels
    A_in = channels_adjustment(A_in)
    W = channels_adjustment(W)

    # m: number of training examples

    # 'in' denotes at input
    # n_w_in (width): number of horizontal pixels of image before convolution.
    # n_h_in (height): number of vertical pixels of image before convolution.
    # n_c_in: number of channels of image before convolution.
    # (n_h_in, n_w_in): shape of image before convolution.
    m, n_h_in, n_w_in, n_c_in = A_in.shape

    # f_w: number of horizontal pixels kernel.
    # f_h: number of vertical pixels in kernel.

    # n_w (width): number of horizontal pixels of image after convolution.
    # n_h (height): number of vertical pixels of image after convolution.
    # n_c: number of channels of image after convolution (number of kernels).
    # (n_h, n_w): shape of image after convolution.
    f_h, f_w, n_c_in, n_c = W.shape

    # Computing the shape of image after convolution.
    n_h = ceil((n_h_in - f_h + 2 * padding) / stride)
    n_w = ceil((n_w_in - f_w + 2 * padding) / stride)

    # Padding image before convolution with zeros
    A_in_padded = add_zero_padding(A_in, padding)

    # Initialize the output image (image after conv.) with zeros
    Z = np.zeros((m, n_h, n_w, n_c))

    for i in range(m):
        a_in_padded = A_in_padded[i]
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c):

                    # getting corner coordinates for a slice for convolution.
                    # vertical coordinates
                    h_start = h * stride
                    h_end = h_start + f_h
                    # horizontal coordinates
                    w_start = w * stride
                    w_end = w_start + f_w

                    # getting the input slice for convolution
                    a_in_slice = a_in_padded[h_start:h_end, w_start:w_end, :]

                    # convolution over input slice
                    Z[i, h, w, c] = convolution_operation(
                        a_in_slice, W[:,:,:, c], b[:,:,:, c])
                    
                    # Applying the activation function.
                    A = G(Z)
    
    """cache = {
        'A_in': A_in,
        'W': W,
        'b': b,
        'padding': padding,
        'stride': stride
    }"""
    cache = (A_in, W, b, padding, stride)    
    return Z, A, cache


def MAX_pool_forward(A_in, pool_size, stride):

    A_in = channels_adjustment(A_in)
    m, n_h_in, n_w_in, n_c_in = A_in.shape

    # Computing the shape of image after convolution.
    n_h = ceil(1 + (n_h_in - pool_size) / stride)
    n_w = ceil(1 + (n_w_in - pool_size) / stride)
    n_c = n_c_in
    
    # Initialize the output image (image after conv.) with zeros
    P = np.zeros((m, n_h, n_w, n_c))             
    
    for i in range(m):
        for h in range(n_h):
            for w in range(n_h):
                for c in range (n_c):
                    
                    # getting corner coordinates for a slice for convolution.
                    # vertical coordinates
                    h_start = h * stride
                    h_end = h_start + pool_size
                    # horizontal coordinates
                    w_start = w * stride
                    w_end = w_start + pool_size

                    # getting the input slice for convolution
                    a_in_slice = A_in[i, h_start:h_end, w_start:w_end, c]

                    # getting max value in output slice matrix
                    P[i, h, w, c] = np.max(a_in_slice)

    # saving intermediate values to be used during backpropagation.
    """cache = {
        'A_in': A_in,
        'pool_size': pool_size,
        'stride': stride
    }"""
    cache = (A_in, pool_size, stride)
    return P, cache


if __name__ == "__main__":
    # Testing function: CNN_node_forward
    from test_fpass import *
    test_cnn_forward_propagation()
