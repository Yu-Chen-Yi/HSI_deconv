''' Functions in deblur flow '''

import numpy as np
from scipy.signal import convolve2d
np.set_printoptions(threshold=np.inf)
import sys
DBL_MIN = sys.float_info.min

########################################################
def kernal_preprocess(img_in, k_in, to_linear, gamma):
    """ kernal_preprocess
            Args:
                img_in (uint8 ndarray, shape(ch, height, width)): Blurred image
                k_in (uint8 ndarray, shape(height, width)): Blur kernel
                to_linear (bool): transform the photo to the linear domian before conducting deblur flow,
                                  and turn back to nonlinear domian after finishing deblur flow
                gamma (float): gamma for transfering linear domain and nonlinear domain

            Returns:
                k_pad (uint8 ndarray, shape(height, width)): Blur kernel after preprocessing
                
            Todo:
                kernal preprocess for Wiener deconvolution
    """
    # Check input shapes
    assert img_in.ndim == 3 and k_in.ndim == 2
    k_in = np.array(k_in, dtype=np.float64)
    # Pad the kernel to match the height and width of img_in
    pad_height = img_in.shape[1] - k_in.shape[0]
    pad_width = img_in.shape[2] - k_in.shape[1]
    # Pad the kernel to match the height and width of img_in
    k_pad = np.pad(k_in, ((0, pad_height), (0, pad_width)), mode='constant')
    k_pad = np.roll(k_pad, -int(k_in.shape[0]//2), axis=0)
    k_pad = np.roll(k_pad, -int(k_in.shape[1]//2), axis=1)
    #k_pad = np.roll(k_pad, -12, axis=0)
    #k_pad = np.roll(k_pad, -12, axis=1)
    if to_linear:
        k_pad = np.power(k_pad/255, gamma)
        k_pad = np.round(np.clip(k_pad, 0, 1)*255)
    return k_pad

########################################################
def deconv_Wiener(img_in, k_in, SNR_F, to_linear, gamma):
    """ Wiener deconvolution
            Args:
                img_in (uint8 ndarray, shape(ch, height, width)): Blurred image
                k_in (uint8 ndarray, shape(height, width)): Padded blur kernel
                SNR_F (float): Wiener deconvolution parameter
                to_linear (bool): transform the photo to the linear domian before conducting deblur flow,
                                  and turn back to nonlinear domian after finishing deblur flow
                gamma (float): gamma for transfering linear domain and nonlinear domain

            Returns:
                Wiener_result (uint8 ndarray, shape(ch, height, width)): Wiener-deconv image
                
            Todo:
                Wiener deconvolution
    """
    # axis=0 is [R,G,B] channel
    # axis=1 is height of image
    # axis=2 is width of image
    # Check input shapes
    assert img_in.ndim == 3 and k_in.ndim == 2

    # Convert to float and normalize to [0, 1]
    img = img_in.astype(np.float64) / 255.0
    k = k_in.astype(np.float64) / 255
    k = k / np.sum(k)

    kernel_fft = np.fft.rfft2(k, axes=(0, 1))
    Wiener_result = np.zeros_like(img, dtype=np.float64)

    wiener_filter = np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + 1 / SNR_F)
    # Transform to linear domain if needed
    if to_linear:
        img = np.power(img, gamma)
    
    # Perform Wiener deconvolution for each channel
    for ch in range(img.shape[0]):
        blurred_fft = np.fft.rfft2(img[ch, :, :]) #B
        filtered_fft = blurred_fft * wiener_filter
        Wiener_result[ch, :, :] = np.abs(np.fft.irfft2(filtered_fft))

    # Transform back to nonlinear domain if needed
    if to_linear:
        Wiener_result = np.power(Wiener_result, 1/gamma)
    # Clip and convert to uint8
    Wiener_result = np.clip(Wiener_result*255, 0, 255)
    Wiener_result = np.round(Wiener_result)
    return Wiener_result.astype(np.uint8)


########################################################
def deconv_RL(img_in, k_in, max_iter, to_linear, gamma):
    """ RL deconvolution
            Args:
                img_in (uint8 ndarray, shape(ch, height, width)): Blurred image
                k_in (uint8 ndarray, shape(height, width)): blur kernel
                max_iter (int): total iteration count
                to_linear (bool): transform the photo to the linear domian before conducting deblur flow,
                                  and turn back to nonlinear domian after finishing deblur flow
                gamma (float): gamma for transfering linear domain and nonlinear domain

            Returns:
                RL_result (uint8 ndarray, shape(ch, height, width)): RL-deblurred image
                
            Todo:
                RL deconvolution
    """
    # Check input shapes
    assert img_in.ndim == 3 and k_in.ndim == 2

    # Convert to float and normalize to [0, 1]
    img = img_in.astype(np.float64) / 255.0
    k = k_in.astype(np.float64) / 255
    
    # Transform to linear domain if needed
    if to_linear:
        img = np.power(img, gamma)
        k = np.power(k, gamma)
        k = k / np.sum(k)
    k_star = k[::-1, ::-1]
    # Perform RL deconvolution for each channel
    RL_result = np.zeros_like(img)
    for ch in range(img.shape[0]):
        B = img[ch]
        I = B.copy()
        for i in range(max_iter):
            # Convolve the current estimate with the kernel
            conv = convolve2d(I, k, mode='same', boundary='symm') + 1e-10

            # Update the estimate
            I *=  convolve2d(B / conv, k_star, mode='same', boundary='symm')
            """ if to_linear:
                I2 = np.power(I, 1/gamma)
            else:
                I2 = I
            filename = f"./ch{ch}_iter{i+1}.png"
            intermediate_image = np.clip(I2, 0, 1)*255
            intermediate_image = np.round(intermediate_image)
            intermediate_image = intermediate_image.astype(np.uint8)
            cv.imwrite(filename, intermediate_image) """
        RL_result[ch] = I

    # Transform back to nonlinear domain if needed
    if to_linear:
        RL_result = np.power(RL_result, 1/gamma)

    # Clip and convert to uint8
    RL_result = np.clip(RL_result, 0, 1)*255
    RL_result = np.round(RL_result)
    
    return RL_result.astype(np.uint8)

########################################################
def BRL_DEB(I_pad, sigma_r, rk):
    """
    Compute the bilateral distance between pixels at (x, y) and all other pixels in the image.
    
    Args:
    I (numpy.ndarray): The image, with shape (height, width).
    sigma_r (float): The standard deviation of the range Gaussian.
    
    Returns:
    numpy.ndarray: The bilateral distance map, with shape (height, width).
    """
    ro = rk//2
    h, w = I_pad.shape
    ep = np.zeros((h-2*ro, w-2*ro))
    x = np.array([ro,ro])
    Ix = I_pad[ro:-ro,ro:-ro]
    for i in range(ro*2+1):
        for j in range(ro*2+1):
            y = np.array([i,j])
            Iy = I_pad[i : (h-2*ro)+i, j : (w-2*ro)+j]
            Spatial = np.exp(- np.sum(np.abs(x - y)** 2) / (2*(ro/3)**2))
            Robust_range = np.exp(- np.abs(Ix - Iy)** 2 / (2 * sigma_r))
            ep +=  Spatial* Robust_range * (Ix - Iy) / sigma_r
    return ep*2

########################################################
def deconv_BRL(img_in, k_in, max_iter, lamb_da, sigma_r, rk, to_linear, gamma):
    """ BRL deconvolution
            Args:
                img_in (uint8 ndarray, shape(ch, height, width)): Blurred image
                k_in (uint8 ndarray, shape(height, width)): Blur kernel
                max_iter (int): Total iteration count
                lamb_da (float): BRL parameter
                sigma_r (float): BRL parameter
                rk (int): BRL parameter
                to_linear (bool): transform the photo to the linear domian before conducting deblur flow,
                                  and turn back to nonlinear domian after finishing deblur flow
                gamma (float): gamma for transfering linear domain and nonlinear domain

            Returns:
                BRL_result (uint8 ndarray, shape(ch, height, width)): BRL-deblurred image
                
            Todo:
                BRL deconvolution
    """
    # Check input shapes
    assert img_in.ndim == 3 and k_in.ndim == 2

    # Convert to float and normalize to [0, 1]
    img = img_in.astype(np.float64) / 255.0
    k = k_in.astype(np.float64) / 255
    
    # Transform to linear domain if needed
    if to_linear:
        img = np.power(img, gamma)
        k = np.power(k, gamma)
        k = k / np.sum(k)
    k_star = k[::-1, ::-1]
    ro = rk//2    
    # Perform RL deconvolution for each channel
    BRL_result = np.zeros_like(img)
    for ch in range(img.shape[0]):
        B = img[ch]
        I = B.copy()
        for i in range(max_iter):
            I_pad = np.pad(I, ((ro, ro), (ro, ro)), mode='symmetric')
            
            # Convolve the current estimate with the kernel
            conv = convolve2d(I, k, mode='same', boundary='symm') + 1e-10

            # EB edge-preserving regularzation term EB(I)
            EB = BRL_DEB(I_pad, sigma_r, rk)

            # Compute the EB (edge-preserving) term
            D_EB = 1 / (1 + lamb_da * EB)
            
            # Update the estimate
            I = I * D_EB  * convolve2d(B / conv, k_star, mode='same', boundary='symm')

            """ if to_linear:
                I2 = np.power(I, 1/gamma)
            else:
                I2 = I
            filename = f"./ch{ch}_iter{i+1}.png"
            intermediate_image = np.clip(I2, 0, 1)*255
            intermediate_image = np.round(intermediate_image)
            intermediate_image = intermediate_image.astype(np.uint8)
            cv.imwrite(filename, intermediate_image) """
        BRL_result[ch] = I

    # Transform back to nonlinear domain if needed
    if to_linear:
        BRL_result = np.power(BRL_result, 1/gamma)

    # Clip and convert to uint8
    BRL_result = np.clip(BRL_result, 0, 1)*255
    BRL_result = np.round(BRL_result)
    return BRL_result.astype(np.uint8)
    
########################################################
def BRL_EB(I_in, sigma_r, rk):
    """ BRL Edge-preserving regularization term
            Args:
                I_in (uint8 ndarray, shape(ch, height, width)): Deblurred image
                sigma_r (float): BRL parameter
                rk (int): BRL parameter
                
            Returns:
                EB (float ndarray, shape(ch)): Edge-preserving regularization term
                
            Todo:
                Calculate BRL Edge-preserving regularization term
    """
    I_in = I_in.astype(np.float64) / 255.0
    EB_CH = np.empty(3)
    ro = rk//2
    for ch in range(I_in.shape[0]):
        I_in_ch = I_in[ch]
        I_pad = np.pad(I_in_ch, ((ro, ro), (ro, ro)), mode='symmetric')
        h, w = I_pad.shape
        EB = np.zeros((h-2*ro, w-2*ro))
        x = np.array([ro,ro])
        Ix = I_pad[ro:-ro,ro:-ro]
        for i in range(ro*2+1):
            for j in range(ro*2+1):
                y = np.array([i,j])
                Iy = I_pad[i : (h-2*ro)+i, j : (w-2*ro)+j]
                Spatial = np.exp(- np.sum(np.abs(x - y)** 2) / (2*(ro/3)**2))
                Robust_range = np.exp(- np.abs(Ix - Iy)** 2 / (2 * sigma_r))
                EB +=  Spatial* (1 - Robust_range)
        EB_CH[ch] = np.sum(EB)
    return EB_CH