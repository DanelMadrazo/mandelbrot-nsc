# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 15:56:56 2026
Author : [ Danel Madrazo ]
Course : Numerical Scientific Computing 2026
"""
import numpy as np
from numba import njit
from multiprocessing import Pool
import time, os, statistics
import matplotlib.pyplot as plt

@njit
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = z_imag = 0.0
    for i in range(max_iter):
        z_sq = z_real*z_real + z_imag*z_imag
        if z_sq > 4.0: return i
        z_imag = 2.0*z_real*z_imag + c_imag
        z_real = z_real*z_real - z_imag*z_imag + c_real
    return max_iter

@njit
def mandelbrot_chunk(row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter):
    out = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for col in range(N):
            out[r, col] = mandelbrot_pixel(x_min + col*dx, c_imag, max_iter)
    return out

def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=100):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)

if __name__ == '__main__':
    N, max_iter = 1024, 100
    x_min, x_max, y_min, y_max = -2, 1, -1.5, 1.5

    t0 = time.perf_counter()
    resultado_serial = mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter)
    t1 = time.perf_counter()
    
    print(f"done in {t1 - t0:.3f} seconds.")
    
    plt.imshow(resultado_serial, cmap='hot')
    plt.show()