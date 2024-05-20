''' Solve deblur problem by Proximal '''
import sys

from proximal.utils.utils import *
from proximal.halide.halide import *
from proximal.lin_ops import *
from proximal.prox_fns import *
from proximal.algorithms import *

import numpy as np
import time


# ########################################################
def TVL1(img_in, k_in, max_iter, lamb_da):
    """ TVL1 deblur
            Args:
                img_in (uint8 ndarray, shape(height, width, ch)): Blurred image
                k_in (uint8 ndarray, shape(height, width)): Blur kernel
                max_iter (int): Total iteration count
                lamb_da (float): TVL1 variable
                
            Returns:
                TVL1_result (uint8 ndarray, shape(height, width, ch)): Deblurred image
                
            Todo:
                TVL1 deblur
    """
    

    img = img_in/255.0
    K = k_in/255.0
    K = K/K.sum()
    
    
    K_rgb = np.zeros((K.shape[0], K.shape[1], 3))
    K_rgb[:,:,0] = K
    K_rgb[:,:,1] = K
    K_rgb[:,:,2] = K
    K = K_rgb
    
    # test the solver with some sparse gradient deconvolution
    eps_abs_rel = 1e-3
    test_solver = 'pc'

    
    #%% rgb channels
    TVL1_result = Variable(img.shape)
    
    # model the problem by proximal |I conv K - B| + lamb_da * |grad I|
    prob = Problem(norm1(conv(K,TVL1_result, dims=2) - img) + lamb_da * group_norm1( grad(TVL1_result, dims = 2), [3] ) + nonneg(TVL1_result)) # formulate problem
    
    # solve the problem
    result = prob.solve(verbose=True,solver=test_solver,x0=img,eps_abs=eps_abs_rel, eps_rel=eps_abs_rel,max_iters=max_iter) # solve problem
    TVL1_result = TVL1_result.value
    
    # output color image
    TVL1_result = np.clip(TVL1_result*255+0.5,0,255).astype('uint8')
    return TVL1_result

########################################################
def TVL2(img_in, k_in, max_iter, lamb_da, to_linear, gamma):
    """ TVL2 deblur
            Args:
                img_in (uint8 ndarray, shape(height, width, ch)): Blurred image
                k_in (uint8 ndarray, shape(height, width)): blur kernel
                max_iter (int): total iteration count
                lamb_da (float): TVL2 variable
                to_linear (bool): transform the photo to the linear domian before conducting deblur flow,
                                  and turn back to nonlinear domian after finishing deblur flow
                gamma (float): gamma for transfering linear domain and nonlinear domain
                
            Returns:
                TVL2_result (uint8 ndarray, shape(height, width, ch)): deblurred image
                
            Todo:
                TVL2 deblur
    """
    img = img_in/255.0
    # Transform to linear domain if required
    if to_linear:
        img = np.power(img, gamma)
    K = k_in/255.0
    K = K/K.sum()
    
    
    K_rgb = np.zeros((K.shape[0], K.shape[1], 3))
    K_rgb[:,:,0] = K
    K_rgb[:,:,1] = K
    K_rgb[:,:,2] = K
    K = K_rgb
    
    # test the solver with some sparse gradient deconvolution
    eps_abs_rel = 1e-3
    test_solver = 'pc'

    
    #%% rgb channels
    TVL1_result = Variable(img.shape)
    
    # model the problem by proximal |I conv K - B| + lamb_da * |grad I|
    prob = Problem(sum_squares(conv(K,TVL1_result, dims=2) - img) + lamb_da * group_norm1( grad(TVL1_result, dims = 2), [3] ) + nonneg(TVL1_result)) # formulate problem
    
    # solve the problem
    result = prob.solve(verbose=True,solver=test_solver,x0=img,eps_abs=eps_abs_rel, eps_rel=eps_abs_rel,max_iters=max_iter) # solve problem
    TVL1_result = TVL1_result.value

    # Transform back to non-linear domain if required
    if to_linear:
        TVL1_result = np.clip(np.power(TVL1_result, 1 / gamma), 0, 1)

    # output color image
    TVL1_result = np.clip(TVL1_result*255+0.5,0,255).astype('uint8')

    return TVL1_result



########################################################
def TVpoisson(img_in, k_in, max_iter, lamb_da, to_linear, gamma):
    """ TVLpoisson deblur
            Args:
                img_in (uint8 ndarray, shape(ch, height, width)): Blurred image
                k_in (uint8 ndarray, shape(height, width)): blur kernel
                max_iter (int): total iteration count
                lamb_da (float): TVpoisson variable
                to_linear (bool): transform the photo to the linear domian before conducting deblur flow,
                                  and turn back to nonlinear domian after finishing deblur flow
                gamma (float): gamma for transfering linear domain and nonlinear domain
                
            Returns:
                TVpoisson_result (uint8 ndarray, shape(ch, height, width)): deblurred image
                
            Todo:
                TVpoisson deblur
    """
    img = img_in/255.0
    # Transform to linear domain if required
    if to_linear:
        img = np.power(img, gamma)
    K = k_in/255.0
    K = K/K.sum()
    
    
    K_rgb = np.zeros((K.shape[0], K.shape[1], 3))
    K_rgb[:,:,0] = K
    K_rgb[:,:,1] = K
    K_rgb[:,:,2] = K
    K = K_rgb
    
    # test the solver with some sparse gradient deconvolution
    eps_abs_rel = 1e-3
    test_solver = 'pc'

    
    #%% rgb channels
    TVL1_result = Variable(img.shape)
    
    # model the problem by proximal |I conv K - B| + lamb_da * |grad I|
    prob = Problem(poisson_norm(conv(K, TVL1_result, dims=2), img) + lamb_da * group_norm1(grad(TVL1_result, dims=2), [3]) + nonneg(TVL1_result))    
    
    # solve the problem
    result = prob.solve(verbose=True,solver=test_solver,x0=img,eps_abs=eps_abs_rel, eps_rel=eps_abs_rel,max_iters=max_iter) # solve problem
    TVL1_result = TVL1_result.value

    # Transform back to non-linear domain if required
    if to_linear:
        TVL1_result = np.clip(np.power(TVL1_result, 1 / gamma), 0, 1)

    # output color image
    TVL1_result = np.clip(TVL1_result*255+0.5,0,255).astype('uint8')

    return TVL1_result