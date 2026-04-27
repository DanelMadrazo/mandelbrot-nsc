# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:40:01 2026
Author : [ Danel Madrazo ]
Course : Numerical Scientific Computing 2026
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

#MILESTONE 1 L08
N, max_iter, tau = 512, 1000, 0.01
x = np.linspace(-0.7530, -0.7490, N)
y = np.linspace( 0.0990,  0.1030, N)

C64 = (x[np.newaxis, :] + 1j * y[:, np.newaxis]).astype(np.complex128)
C32 = C64.astype(np.complex64)

z32 = np.zeros_like(C32)
z64 = np.zeros_like(C64)

diverge = np.full((N, N), max_iter, dtype=np.int32)
active = np.ones((N, N), dtype=bool)

for k in range(max_iter):
    if not active.any(): break 
    z32[active] = z32[active]**2 + C32[active]
    z64[active] = z64[active]**2 + C64[active]
    diff = (np.abs(z32.real.astype(np.float64) - z64.real) + np.abs(z32.imag.astype(np.float64) - z64.imag))
    newly = active & (diff > tau)
    diverge[newly] = k
    active[newly] = False
plt.imshow(diverge, cmap= 'plasma', origin = 'lower', extent=[-0.7530, -0.7490, 0.0990, 0.1030])
plt.colorbar(label= 'Firts Divergence Iteration')
plt.title(f'Trajectory divergence (tau={tau})')
plt.show()

#MILESTONE 2 L08
C = C64
eps32 = float(np.finfo(np.float32).eps)
delta = np.maximum(eps32 * np.abs(C), 1e-10)
def escape_count(C, max_iter):
    z = np.zeros_like(C)
    cnt = np.full(C.shape, max_iter, dtype=np.int32)
    esc = np.zeros(C.shape, dtype = bool)
    for k in range (max_iter):
        active = ~esc 
        z[active] = z[active]**2 + C[active]
        newly = active & (np.abs(z)>2.0)
        cnt[newly] = k
        esc[newly] = True
    return cnt
n_base = escape_count(C, max_iter).astype(float)
n_perturb = escape_count(C+delta, max_iter).astype(float)
dn = np.abs(n_base-n_perturb)
kappa = np.where(n_base>0, dn/(eps32*n_base), np.nan)
cmap_k = plt.cm.hot.copy()
cmap_k.set_bad('0.25')
vmax = np.nanpercentile(kappa, 99)
plt.imshow(kappa, cmap = cmap_k, origin = 'lower', extent = [-0.7530, -0.7490, 0.0990, 0.1030], norm = LogNorm(vmin=1, vmax=vmax))
plt.colorbar(label=r'$\kappa(c)$ (log scale, $\kappa \geq 1$')
plt.title(r'Condition number approx $\kappa(c) = |\Delta n|\, /\, (\varepsilon_{32}\, n(c))$')
plt.show()


# ==========================================
# L10 MILESTONE 3: GPU Benchmark Comparison
# ==========================================

time_seconds = {
    "Naive Python": 9.0125,       
    "NumPy Vectorizado": 2.2220,  
    "Numba (MP1)": 0.0779,        
    "Multiprocessing (MP2)": 0.0249,
    "Dask Local": 0.1642,
    "GPU OpenCL f32 (M1)": 0.005,
    "GPU f64": 0.0
    }
names = [k for k, v in time_seconds.items() if v > 0.0]
times = [v for k, v in time_seconds.items() if v > 0.0]

plt.figure(figsize=(10, 6))
plt.bar(names, times, log=True, color='#2c7bb6', edgecolor='black')

plt.ylabel("Execution time in seconds (Logaritmic scale)")
plt.title("Mandelbrot Benchmark (N=1024) - CPU vs GPU")
plt.xticks(rotation=30, ha="right")
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(times):
    plt.text(i, v, f"{v:.4f}s", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig("benchmark_mp3.png", dpi=150)
plt.show()