# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:48:35 2026

@author: damad
"""
import math, random, time, statistics, os
from multiprocessing import Pool
import matplotlib.pyplot as plt

def estimate_pi_serial(num_samples):
    inside_circle = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inside_circle += 1
    return 4 * inside_circle / num_samples

def estimate_pi_chunk(num_samples):
    inside_circle = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inside_circle += 1
    return inside_circle
    
def estimate_pi_parallel(num_samples, num_processes=4):
    samples_per_process = num_samples // num_processes
    tasks = [samples_per_process] * num_processes
    with Pool(processes=num_processes) as pool:
        results = pool.map(estimate_pi_chunk, tasks)
    return 4 * sum(results) / num_samples




if __name__ == '__main__':
    num_samples = 10000000
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        pi_estimate = estimate_pi_serial(num_samples)
        times.append(time.perf_counter() - t0)
    t_serial = statistics.median(times)
    print(f"=== Baseline Serial ===")
    print(f"pi estimate: {pi_estimate:.6f} (error: {abs(pi_estimate-math.pi):.6f})")
    print(f"Serial time: {t_serial:.3f}s")
    
    max_speedup = 0
    best_p = 1
    
    workers_list = []
    speedup_list = []
    
    print("workers | time (s) | speedup Sp | efficiency Ep (%)")
    print("-" * 55)
    
    for num_proc in range(1, os.cpu_count() + 1):
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            pi_est = estimate_pi_parallel(num_samples, num_proc)
            times.append(time.perf_counter() - t0)
        t_par = statistics.median(times)
        #print(f"{num_proc:2d} workers:{t_par:.3f}s pi={pi_est:.6f}") 
        
        #E3
        speedup = t_serial / t_par
        efficiency = (speedup / num_proc) * 100
        
        workers_list.append(num_proc)
        speedup_list.append(speedup)
        
        print(f"{num_proc:7d} | {t_par:8.3f} | {speedup:10.2f}x | {efficiency:14.0f}%")
        
        if speedup > max_speedup:
            max_speedup = speedup
            best_p = num_proc
            
    print(f"Maximum speedup (Sp*) = {max_speedup:.2f}x at p* = {best_p} workers")
    
    if best_p > 1:
        s = (1 / max_speedup - 1 / best_p) / (1 - 1 / best_p)
        print(f"Implied serial fraction (s) = {s * 100:.2f}%")
    else:
        print("1 Worker was the best, cannot calculate 's'.")
        
    plt.figure(figsize=(8, 5))
    plt.plot(workers_list, speedup_list, marker='o', linestyle='-', color='steelblue', label='Measured speedup')
    plt.plot([1, os.cpu_count()], [1, os.cpu_count()], linestyle='--', color='lightgray', label='Ideal (linear)')
    plt.axvline(x=os.cpu_count(), linestyle=':', color='gray', alpha=0.5, label=f'Logical cores ({os.cpu_count()})')
    
    plt.title('Monte Carlo $\pi$ speedup (10,000,000 samples)')
    plt.xlabel('Number of worker processes')
    plt.ylabel('Speedup (relative to serial)')
    plt.xticks(workers_list)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    

