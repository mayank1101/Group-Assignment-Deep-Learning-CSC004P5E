import numpy as np
import os
import sys
import inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
import cnn_forward_propagation
from cnn_forward_propagation.fpass import *

def test_cnn_forward_propagation():
    # Testing function: CNN_node_forward
    
    np.random.seed(0)
    data_size = np.random.randint(10, 50)
    
    # dimensions of the convoluted image
    val2 = np.random.randint(1, 4)
    f_h, f_w = val2, val2   # m2, n2
    
    # input image dimensions
    val1 = np.random.randint(4, 12)
    n_h_in, n_w_in = f_h + val1, f_w + val1   # m1, n1
    
    n_c_in = np.random.randint(1, 10)
    n_c = n_c_in + np.random.randint(1, 10)  # Number of kernels
    
    padding, stride = 1, 2
    
    # Initializing input dataset
    A_in = np.random.randn(data_size, n_h_in, n_w_in, n_c_in)
    
    # Initializing weights
    W = np.random.randn(f_h, f_w, n_c_in, n_c)
    b = np.random.randn(1, 1, 1, n_c)
    
    print(f'A_in: {A_in.shape}')
    
    Z, A1, cache_conv1 = CNN_node_forward(A_in, W, b, padding, stride, relu)
    
    print(f'Z: {Z.shape}')
    print(f'A: {A1.shape}')
    # print(cache_conv.keys())
    
    A_prev, W, b, pad, stide = cache_conv1
    
    # Testing MAX_pool_forward
    pool_size = 2
    stride_pool = 2
    P1, cache_pool1 = MAX_pool_forward(A1, pool_size, stride_pool)
    
    print(f'P1: {P1.shape}')
    
    size = P1.shape
    
    return Z, A1, size, cache_conv1, cache_pool1
