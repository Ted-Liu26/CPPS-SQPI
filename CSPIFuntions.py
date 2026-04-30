#!/usr/bin/env python3
"""
 8. 4-step Phase modulation Hadamard reconstruction denoise functions:
    WHRecover4PMDenoise
    WHRecoverAngRnoise
"""

__author__ = "Su-Heng Zhang"
__date__ = "2023-04-11"

import numpy as np
from numpy import fft
from scipy.linalg import hadamard

###############################################################################
# 3. walsh hadamard matrix
###############################################################################
def WHadamard(n):
    if n & (n - 1) != 0:
        print(f"The order must be a power of 2!")
        return []

    ind = [0] * n
    N = len(bin(n)) - 3

    for i in range(n):
        g = i ^ (i >> 1)
        g = bin(g)[2:]
        g = (N - len(g)) * "0" + g
        gr = g[-1::-1]
        ind[i] = int(gr, 2)

    H = hadamard(n)
    return H[ind, :]

###############################################################################
# 8. 4-step Phase modulation walsh hadamard reconstruction denoise
###############################################################################
def WHRecoverPMDN(mdata, N, fb=2):

    # Walsh Hadamard Matrix
    whmatrix = WHadamard(N)
    #whmatrix = np.rot90(whmatrix,k=-1)
    rec_image = np.zeros((N, N))

    u = np.arange(N).reshape(-1, 1)  # spectral coordinates
    v = np.arange(N).reshape(1, -1)
    ind = u * v  # coordinate product as frequency measure

    snoise = np.zeros((N, N))  # initial noise spectral matrix



    had_spec = mdata[:].reshape(N, N)

    # recover image include background noise
    nimage = whmatrix @ had_spec @ whmatrix / (N**2)

    # extract background noise spectrum
    snoise[ind <= fb] = nimage[ind <= fb]
    # get background noise estimation
    cnoise = whmatrix @ snoise @ whmatrix

    # remove background noise from hadamard spectrum
    chspec = had_spec - cnoise

    # recover noise clear image
    cimage = whmatrix @ chspec @ whmatrix / (N**2)

    rec_image = cimage

    return rec_image

def WHRecoverAngRnoise(angle_image, N, fb=10):

    u = np.arange(N).reshape(-1, 1)  # spectral coordinates
    v = np.arange(N).reshape(1, -1)
    ind = u * v  # coordinate product as frequency measure

    renoise = np.zeros((N, N))  # initial noise spectral matrix
    renoise[ind > fb] = angle_image[ind > fb]
    return renoise

