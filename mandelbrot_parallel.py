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
    
    # 1. Warm-up 
    _ = mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter)
    
    # Serial baseline (Numba already warm after M1 warm-up)
    times_serial = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter)
        times_serial.append(time.perf_counter() - t0)
    t_serial = statistics.median(times_serial)
    
    print(f"Serial (baseline): {t_serial:.4f} seconds\n")
    print("--- Parallel Benchmark (M3) ---")
    print("workers | time (s) | speedup Sp | efficiency Ep (%)")
    print("-" * 55)
    
    max_speedup = 0
    best_p = 1
    
    for n_workers in range(1, os.cpu_count() + 1):

        chunk_size = max(1, N // n_workers)
        chunks, row = [], 0
        while row < N:
            end = min(row + chunk_size, N)
            chunks.append((row, end, N, x_min, x_max, y_min, y_max, max_iter))
            row = end
            
        with Pool(processes=n_workers) as pool:
            pool.map(_worker, chunks) # warm-up: Numba JIT in all workers
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                np.vstack(pool.map(_worker, chunks))
                times.append(time.perf_counter() - t0)
        t_par = statistics.median(times)
        speedup = t_serial / t_par
        print(f"{n_workers:2d} workers: {t_par:.3f}s, speedup={speedup:.2f}x, eff={speedup/n_workers*100:.0f}%")
        
        if speedup > max_speedup:
            max_speedup = speedup
            best_p = n_workers
            
    print("-" * 55)
    print("\n=== Amdahl Analisis ===")
    print(f"Maximum speedup (Sp*) = {max_speedup:.2f}x at p* = {best_p} workers")
    
    if best_p > 1:
        # back-solve implied serial fraction (s)
        s = (1 / max_speedup - 1 / best_p) / (1 - 1 / best_p)
        print(f"Implied serial fraction (s) = {s * 100:.2f}%")
    