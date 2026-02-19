"""
Mandelbrot Set Generator
Author : [ Danel Madrazo ]
Course : Numerical Scientific Computing 2026
"""
import numpy as np
import time , statistics
import matplotlib.pyplot as plt

#STEP 2
def mandelbrot_point(c, max_iter = 100):
    z = 0j
    for n in range(max_iter):
        z = z**2 + c 
        if abs(z) > 2:
            return n    
    return max_iter

#STEP 3
def compute_mandelbrot(C, max_iter = 100):
    Z = np.zeros_like(C)
    M = np.zeros(C.shape, dtype=int)
    for i in range(max_iter):  
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask]**2 + C[mask]
        M[mask] += 1
    return M

xmin, xmax, ymin, ymax = -2, 1, -1.5, 1.5
res_x, res_y = 1024, 1024

x = np . linspace ( xmin , xmax, res_x) # 1024 x- values
y = np . linspace ( ymin , ymax , res_y) # 1024 y- values
X , Y = np . meshgrid (x , y) # 2D grids
C = X + 1j* Y # Complex grid
print (f" Shape : {C. shape }") # (1024 , 1024)
print (f" Type : {C. dtype }") # complex128

# #STEP 4
# start = time.time()
# result = compute_mandelbrot(xmin, xmax, ymin, ymax, res_x, res_y)
# elapsed = time.time() - start

#1024x1024 resolution takes around 4 seconds
#2048x2048 resolution takes around 17 seconds

# print(f"Computation took {elapsed:.3f} seconds")

#Lecture 2, new time measurement function
def benchmark ( func , * args , n_runs =3) :
    """ Time func , return median of n_runs . """
    times = []
    for _ in range ( n_runs ):
        t0 = time.perf_counter()
        result = func (* args)
        times.append(time.perf_counter() - t0 )
    median_t = statistics.median(times)
    print (f" Median : { median_t:.4f}s "
    f"( min ={ min( times ):.4f}, max ={ max( times ):.4f})")
    return median_t , result
t , M = benchmark (compute_mandelbrot, C, 100)

#STEP 5
plt.figure()
plt.imshow(M, cmap = 'viridis')
plt.title("Mandelbrot")
plt.xlabel("Real (Re)")
plt.ylabel("Imaginary (Im)")
plt.show()
        