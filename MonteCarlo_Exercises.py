# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:48:35 2026

@author: damad
"""
import math, random, time, statistics, os
from multiprocessing import Pool
import matplotlib.pyplot as plt
from functools import reduce
from dask.distributed import Client, LocalCluster
import dask
from dask import delayed

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

def monte_carlo_chunk(num_samples): 
    """Estimate pi contributions for num_samples random points."""
    inside = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inside += 1
    return inside

def test_granularity(total_work, chunk_size, n_proc):
    n_chunks = total_work // chunk_size
    tasks = [chunk_size] * n_chunks
    t0 = time.perf_counter()
    if n_proc == 1:
        results = [monte_carlo_chunk(s) for s in tasks]
    else:
        with Pool(processes=n_proc) as pool:
            results = pool.map(monte_carlo_chunk, tasks)
    return time.perf_counter() - t0, 4 * sum(results) / total_work

def subtract_seven(x):
    return x - 7

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
    
    
    # EXERCISE1 - Parallel Computing part 2 - Chunk Size Investigation
    print('CHUNK SIZE INVESTIGATION:')
    total_work = 1_000_000
    n_proc = os.cpu_count() // 2
    chunk_sizes = [10, 100, 1_000, 10_000, 100_000, 1_000_000]
    print(f"{'L':>12} | {'serial(s)':>12} | {'parallel(s)':>12}")
    for L in chunk_sizes:
        t_ser, _ = test_granularity(total_work, L, n_proc=1)
        t_par, pi = test_granularity(total_work, L, n_proc=n_proc)
        print(f"{L:12d} | {t_ser:12.4f} | {t_par:12.4f} pi={pi:.4f}")
        
        
    #EXERCISE2 - Parallel Computing part 2 (Lecture 5) - Map-Filter-Reduce
    N = 1_000_000
    data = [random.randint(10, 100) for _ in range(N)]
    
    #Part 1
    t0 = time.perf_counter()
    result_ser = reduce(lambda a, b: a + b, filter(lambda x: x % 2 == 1, map(subtract_seven, data)))
    t_serial = time.perf_counter() - t0
    
    #Part 2
    t0 = time.perf_counter()
    with Pool() as pool:
        mapped = pool.map(subtract_seven, data)
    result_par = reduce(lambda a, b: a + b, filter(lambda x: x % 2 == 1, mapped))
    t_parallel = time.perf_counter() - t0
    print(f"Serial: {t_serial:.4f}s result={result_ser}")
    print(f"Parallel: {t_parallel:.4f}s result={result_par}")
    print(f"Speedup: {t_serial / t_parallel:.2f}x")
    
    #EXERCISE1 - Lecture 6 - Dask Delayed — Lazy Evaluation
    total, n_chunks = 1_000_000, 8
    samples = total // n_chunks # Serial baseline
    t0 = time.perf_counter()
    results = [monte_carlo_chunk(samples) for
    _ in range(n_chunks)]
    t_serial = time.perf_counter() - t0
    print(f"Serial:{t_serial:.3f}s pi={4*sum(results)/total:.4f}")
    # Dask delayed -- task graph is built, not executed yet
    tasks = [delayed(monte_carlo_chunk)(samples) for
    _ in range(n_chunks)]
    t0 = time.perf_counter()
    results = dask.compute(*tasks)
    t_dask = time.perf_counter() - t0
    print(f"Dask:{t_dask:.3f}s pi={4*sum(results)/total:.4f}")
    # Visualise (requires: conda install python-graphviz)
    #dask.visualize(*tasks, filename='task_graph.png')


    #EXERCISE2 - Lecture 6 - LocalCluster & Dashboard
    total, n_chunks = 1_000_000, 8
    samples = total // n_chunks
    max_workers = os.cpu_count()
    # Create local cluster; start with max workers -- scale() adjusts without restarting
    cluster = LocalCluster(n_workers=max_workers, threads_per_worker=1)
    client = Client(cluster)
    print(f"Dashboard: {client.dashboard_link}")
    # --> open the printed URL in your browser
    input("Open the URL in your browser, look at the Task Stream tab, then press Enter...")
    
    
    # Rerun E1 tasks; LocalCluster scheduler takes over
    print(f"\nRunning with {max_workers} workers...")
    t0 = time.perf_counter()
    tasks = [delayed(monte_carlo_chunk)(samples) for _ in range(n_chunks)]
    results = dask.compute(*tasks)
    t_dask_full = time.perf_counter() - t0
    print(f"Time ({max_workers} workers): {t_dask_full:.3f}s | pi={4*sum(results)/total:.4f}")
    
    
    # Vary n_workers: scale() resizes without restarting the scheduler
    # (recreating LocalCluster while the browser is open breaks the dashboard)
    half_workers = max(1, max_workers // 2)
    print(f"\nScaling cluster down to {half_workers} workers...")
    
    cluster.scale(half_workers) 
    #client.wait_for_workers(half_workers)
    time.sleep(2) # Give Windows a moment to safely kill the processes
    
    t0 = time.perf_counter()
    tasks = [delayed(monte_carlo_chunk)(samples) for _ in range(n_chunks)]
    results = dask.compute(*tasks)
    t_dask_scaled = time.perf_counter() - t0
    print(f"Time ({half_workers} workers): {t_dask_scaled:.3f}s | pi={4*sum(results)/total:.4f}")
    
    client.close(); cluster.close()
    cluster.close()
