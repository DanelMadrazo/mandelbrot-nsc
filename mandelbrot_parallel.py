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
        zr2 = z_real*z_real
        zi2 = z_imag*z_imag
        if zr2 + zi2 > 4.0: return i
        z_imag = 2.0*z_real*z_imag + c_imag
        z_real = zr2 - zi2 + c_real
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

def _worker(args):
    return mandelbrot_chunk(*args)

def mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter=100, n_workers=4):
    chunk_size = max(1, N // n_workers)
    chunks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
    with Pool(processes=n_workers) as pool:
        pool.map(_worker, chunks) # un-timed warm-up: Numba JIT in workers
        parts = pool.map(_worker, chunks)
    return np.vstack(parts)


if __name__ == '__main__':
    N, max_iter = 1024, 100
    x_min, x_max, y_min, y_max = -2, 1, -1.5, 1.5
        

    t0_serial = time.perf_counter()
    result_serial = mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter)
    t1_serial = time.perf_counter()
    
    print(f"serial done in {t1_serial - t0_serial:.3f} seconds.")
    
    plt.imshow(result_serial, cmap='hot')
    plt.show()
    
    t0_parallel = time.perf_counter()
    result_parallel = mandelbrot_parallel(N, x_min, x_max, y_min, y_max, n_workers=4)
    t1_parallel = time.perf_counter()
    
    print(f"parallel done in {t1_parallel - t0_parallel:.3f} seconds.")
    plt.imshow(result_parallel, cmap='hot')
    plt.show()
    
    if np.array_equal(result_serial, result_parallel):
        print("Yes, parallel result is exactly the same as serial")
    else:
        print("No, there is some differences between serial and parallel results")