import os
import sys
import inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
import cnn_forward_propagation
from cnn_forward_propagation.test_fpass import *
import numpy as np
from bpass import *

def test_cnn_backward_propagation():

    np.random.seed(1)
    Z, A1, size, cache_conv1, cache_pool1 = test_cnn_forward_propagation()
    dA,dW,db = CNN_node_backward(Z,cache_conv1)
    print("\ndA_mean =", np.mean(dA))
    print("dW_mean =", np.mean(dW))
    print("db_mean =", np.mean(db))

    dA = np.random.randn(*size)
    dA_prev = pool_backward(dA, cache_pool1)

    print('\nmean of dA = ', np.mean(dA))
    print('dA_prev[1,1] = ', dA_prev[1,1])  
    print()