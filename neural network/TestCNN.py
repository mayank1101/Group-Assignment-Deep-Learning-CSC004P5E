import numpy as np
from CNN.py import ConvolutionNN

def TestForwardPass():

    np.random.seed(0)
    data_size = np.random.randint(10, 50)
    
    for i in range(2,6):

        f_h, f_w = i, i # Filter height and width (m2, n2)
        padding, stride = i, i
        print('For filter with size : ({}, {})'.format(f_h, f_w), ' Padding :',padding, ' Stride :', stride, '\n')

        # input image dimensions
        for j in range(5, 12):
            val = np.random.randint(4, j)
            n_h_in, n_w_in = f_h + val, f_w + val   # m1, n1

            n_c_in = np.random.randint(1, 10)
            n_c = n_c_in + np.random.randint(1, 10)  # Number of kernels

            # Initializing input dataset
            A_in = np.random.randn(data_size, n_h_in, n_w_in, n_c_in)

            # Initializing weights
            W = np.random.randn(f_h, f_w, n_c_in, n_c)
            b = np.random.randn(1, 1, 1, n_c)
        
            Z, A1, cache_conv1 = CNN_node_forward(A_in, W, b, padding, stride, relu)

            pool_size, stride_pool = 2, 2
            P1, cache_pool1 = MAX_pool_forward(A1, pool_size, stride_pool)
            
            print('Input Image Dimesions :',A_in.shape,' Convoluted Image Dimensions :',A1.shape,' Image After Pooling :',P1.shape)
        print()
        
TestForwardPass()