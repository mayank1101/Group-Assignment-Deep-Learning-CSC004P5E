from math import ceil
import numpy as np
import os
import sys
import inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
import cnn_forward_propagation
from cnn_forward_propagation.fpass import add_zero_padding


def  CNN_node_backward(dZ, cache):

  """
  Inputs :
  dZ : gradient of the cost with respect to the out of convolution layer(Z) of shape(m,n_H,n_W,n_C)
  cache : values needed for backward convolution and out of forward convolution

  Returns :
  dA_prev : gradient of cost w.r.t the i/p of convolution layer
  dW: gradient of the cost w.r.t weights of convolution layer
  db :gradient of the cost w.r.t biases of convolution layer
  """

  
  # Retrivieving values from the cache saved while forward pass

  (Aprev_layer_output, prev_layer_kernel_Weights, prev_layer_bias, padding, stride ) = cache

  # Dimensions of the Aprev_layer_output shape
  '''
  m : num of training examples
  n_h_prev = number of rows(verticle pixel) in prev layer
  n_w_prev = number of columns(horizontal pixel) in prev layer
  n_c_prev = number of channels in prev layer

  batches considered in psudo code is actually number of inputs
  '''

  (m, n_h_prev, n_w_prev, n_c_prev) = Aprev_layer_output.shape

  # retrieving shape from the prev_layer_kernel_Weights
  # f_h :  horizontal pixels in kernel
  # f_w =  verticle pixels in kernel
  # n_c = no of channels

  (f_h,f_w,n_c_prev,n_c) = prev_layer_kernel_Weights.shape

  (m, n_h,n_w, n_c) = dZ.shape

  #Initializing the dA_prev, dW, db 
  dA_prev = np.zeros((m,n_h_prev, n_w_prev, n_c_prev))
  dW = np.zeros((f_h,f_h,n_c_prev,n_c))
  db = np.zeros((1,1,1,n_c))

  A_prev_pad = add_zero_padding(Aprev_layer_output,padding)
  dA_prev_pad = add_zero_padding(dA_prev, padding)

  for i in range(m):                          # number of batches/ i.e. number of elements
    a_prev_pad = A_prev_pad[i]
    da_prev_pad = dA_prev_pad[i]

    for h in range(n_h):         # loop over verticle axis of output
     for w in range(n_w):        # loop over horizontal axis of output
       for c in range(n_c):
         
         vert_start = h * stride
         vert_end = vert_start + f_h
         horiz_start = w * stride
         horiz_end = horiz_start + f_h

         a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end,:]

         da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += prev_layer_kernel_Weights[:, :, :, c] * dZ[i, h, w, c]
         
         #da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += prev_layer_kernel_Weights[:,;,:,c] * dZ[i,h,w,c]

         dW[:,:,:,c] += a_slice * dZ[i,h,w,c]
         db[:,:,:,c] += dZ[i,w,h,c]


     dA_prev[i,:,:,:] = da_prev_pad[padding:-padding, padding:-padding,:]


     return dA_prev, dW, db

def max_pool_mask(x):
  mask = np.max(x) == x

  return mask

def pool_backward(dA, cache):

  """
  Inputs:

    dA = gradient of cost w.r.t output of pooling layer
         shape same as A
    cahce = cahce output from the forward pass of pooling layer,
            layers input and hyperpara

    Returns:

    dA_prev = gradient of cost w.r.t
  """


  (A_prev, pool_size, stride ) = cache

  m, n_h_prev, n_w_prev, n_c_prev = A_prev.shape
  m, n_h, n_w, n_c = dA.shape

  dA_prev = np.zeros((m,n_h_prev,n_w_prev,n_c_prev))

  for i in range(m):
    a_prev = A_prev[i]

    for h in range(n_h):
      for w in range(n_w):
        for c in range(n_c):

          vert_start = h * stride
          vert_end = vert_start + pool_size
          horiz_start = w * stride
          horiz_end = horiz_start + pool_size

          a_prev_slice = a_prev[vert_start:vert_end,horiz_start:horiz_end,c] 

          mask = max_pool_mask(a_prev_slice)

          dA_prev[i,vert_start: vert_end, horiz_start: horiz_end, c] += np.multiply(mask, dA[i, h, w, c]) 
  
  return dA_prev


if __name__ == '__main__':
    from test_bpass import *
    test_cnn_backward_propagation()
