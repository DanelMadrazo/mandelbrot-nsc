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

@njit(cache=True)
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = z_imag = 0.0
    for i in range(max_iter):
        zr2 = z_real*z_real
        zi2 = z_imag*z_imag
        if zr2 + zi2 > 4.0: return i
        z_imag = 2.0*z_real*z_imag + c_imag
        z_real = zr2 - zi2 + c_real
    return max_iter

@njit(cache=True)
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

def mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter=100, n_workers=4, n_chunks=None, pool= None):
    if n_chunks is None:
        n_chunks = n_workers
    chunk_size = max(1, N // n_chunks)
    chunks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
        
    if pool is not None: # caller manages Pool; skip startup + warm-up
        return np.vstack(pool.map(_worker, chunks))
    tiny = [(0, 8, 8, x_min, x_max, y_min, y_max, max_iter)]
    
    with Pool(processes=n_workers) as pool:
        pool.map(_worker, tiny) # warm-up: load JIT cache in workers
        parts = pool.map(_worker, chunks)
    return np.vstack(parts)

#Next ones are necessary for comparison in M3 MP2 of lecture5
def mandelbrot_point(c, max_iter = 100):
    z = 0j
    for n in range(max_iter):
        z = z**2 + c 
        if abs(z) > 2:
            return n    
    return max_iter
def compute_mandelbrot_naive(xmin, xmax, ymin, ymax, x_res, y_res, max_iter = 100):
    x = np.linspace(xmin, xmax, x_res)
    y = np.linspace(ymin, ymax, y_res)

    iteration_num = np.zeros((y_res, x_res))

    for i in range(y_res):
        for j in range(x_res):
            c = complex(x[j], y[i])
            n = mandelbrot_point(c, max_iter)
            iteration_num[i, j] = n
    return iteration_num
def compute_mandelbrot_numpy(C, max_iter = 100):
    Z = np.zeros_like(C)
    M = np.zeros(C.shape, dtype=int)
    for i in range(max_iter):  
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask]**2 + C[mask]
        M[mask] += 1
    return M

if __name__ == '__main__':
    
    #LECTURE 4:
        
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
    
    workers_list = []
    speedup_list = []
    
    max_workers = os.cpu_count()
    
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
        
        workers_list.append(n_workers)
        speedup_list.append(speedup)
        
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
    
    #SPEEDUP CURVES
    plt.figure()
    
    #Measured speedup
    plt.plot(workers_list, speedup_list, marker='o', linestyle='-', color='steelblue', label='Measured speedup')
    
    #Ideal speedup
    plt.plot([1, max_workers], [1, max_workers], linestyle='--', color='lightgray', label='Ideal (linear)')
    
    #Logical cores vertical line
    plt.axvline(x=max_workers, linestyle=':', color='red', alpha=0.5, label=f'Logical cores ({max_workers})')
    
    #Peak point
    label_text = f"(peak: {max_speedup:.2f}x, {best_p} workers)"
    
    plt.annotate(
        label_text,                       # Text
        xy=(best_p, max_speedup),         # Coordinates of the point
        xytext=(best_p - 1.5, max_speedup + 0.3), # Where to write the text           
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', lw=0.5), 
        arrowprops=dict(arrowstyle='-', connectionstyle='arc3', color='black', alpha=0.5)
    )
    
    plt.title(f'Parallel Mandelbrot speedup (N={N}, max_iter={max_iter}, 3 runs)')
    plt.xlabel('Number of worker processes')
    plt.ylabel('Speedup (relative to serial)')
    plt.xticks(workers_list)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    #LECTURE 5:
    #Milestone 1
    print("\n" + "="*55)
    print("--- M1: Verification (n_chunks=32) ---")
    
    ref_result = mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter)
    test_result = mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter, 
                                         n_workers=max_workers, n_chunks=32)
    if np.array_equal(ref_result, test_result):
        print('Yes, they are equal')
    else:
        print('No, there are differences')
        
    #Milestone 2
    print("\n" + "="*55)
    print(f"--- M2: Mandelbrot Granularity Sweep (Workers = {max_workers}) ---")
    print(" chunks | time (s) | speedup Sp | LIF")
    print("-" * 55)
    
    chunk_counts = [m * max_workers for m in [1, 2, 4, 8, 16]]
    
    max_speedup_l5 = 0
    best_chunks = 1
    speedups_l5 = []
    
    with Pool(processes=max_workers) as pool:
        tiny_chunk = [(0, 8, N, x_min, x_max, y_min, y_max, max_iter)]
        pool.map(_worker, tiny_chunk)
        
        for n_chunks in chunk_counts:
            times_par = []
            for _ in range(3):
                t0 = time.perf_counter()
                _ = mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter, 
                                        n_workers=max_workers, n_chunks=n_chunks, pool=pool)
                times_par.append(time.perf_counter() - t0)
                
            t_par = statistics.median(times_par)
            speedup = t_serial / t_par
            speedups_l5.append(speedup)
            
            # Calculate Load Imbalance Factor (LIF)
            lif = max_workers * (t_par / t_serial) - 1
            
            print(f"{n_chunks:7d} | {t_par:8.4f} | {speedup:8.2f}x | {lif:5.2f}")
            
            if speedup > max_speedup_l5:
                max_speedup_l5 = speedup
                best_chunks = n_chunks
    
    print("-" * 55)
    print(f"\n=> OPTIMAL L5: {max_speedup_l5:.2f}x speedup using {best_chunks} chunks.")
    
    #Milestone 3
    print("\n" + "="*55)
    print("--- M3: Comprehensive Analysis (1024x1024) ---")
    print("Implementation     | Time (s) | Speedup ")
    print("-" * 55)
    
    #Naive
    t_naive_times = []
    for _ in range(3):
        t0 = time.perf_counter()
        compute_mandelbrot_naive(x_min, x_max, y_min, y_max, N, N, max_iter)
        t_naive_times.append(time.perf_counter() - t0)
    t_naive = statistics.median(t_naive_times)
    print(f"Naive Python       | {t_naive:8.4f} | 1.00x")
    
    #Numpy
    x = np.linspace(x_min, x_max, N)
    y = np.linspace(y_min, y_max, N)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    t_numpy_times = []
    for _ in range(3):
        t0 = time.perf_counter()
        compute_mandelbrot_numpy(C, max_iter)
        t_numpy_times.append(time.perf_counter() - t0)
    t_num = statistics.median(t_numpy_times)
    print(f"NumPy Vectorized   | {t_num:8.4f} | {t_naive / t_num:8.2f}x")
    
    #Numba (@njit) --> (t_serial from earlier)
    print(f"Numba (@njit)      | {t_serial:8.4f} | {t_naive / t_serial:8.2f}x")
    
    #Parallel (run using the best configuration found in M2)
    with Pool(processes=max_workers) as pool:
        pool.map(_worker, tiny_chunk) # Warm-up
        t_opt_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            _ = mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter, 
                                    n_workers=max_workers, n_chunks=best_chunks, pool=pool)
            t_opt_times.append(time.perf_counter() - t0)
    t_opt = statistics.median(t_opt_times)
    print(f"Parallel (opt.)    | {t_opt:8.4f} | {t_naive / t_opt:8.2f}x")
    print("-" * 55)