"""
Mandelbrot Set Generator
Author : [ Danel Madrazo ]
Course : Numerical Scientific Computing 2026
"""
import numpy as np
import time
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
def compute_mandelbrot(xmin, xmax, ymin, ymax, x_res, y_res, max_iter = 100):
    x = np.linspace(xmin, xmax, x_res)
    y = np.linspace(ymin, ymax, y_res)
    
    iteration_num = np.zeros((y_res, x_res))
    
    for i in range(y_res):
        for j in range(x_res):
            c = complex(x[j], y[i])
            n = mandelbrot_point(c, max_iter)
            iteration_num[i, j] = n
    return iteration_num

xmin, xmax, ymin, ymax = -2, 1, -1.5, 1.5
res_x, res_y = 2048, 2048

#STEP 4
start = time.time()
result = compute_mandelbrot(xmin, xmax, ymin, ymax, res_x, res_y)
elapsed = time.time() - start

#1024x1024 resolution takes around 4 seconds
#2048x2048 resolution takes around 17 seconds

print(f"Computation took {elapsed:.3f} seconds")

#STEP 5
plt.figure()
plt.imshow(result, cmap = 'viridis')
plt.title("Mandelbrot")
plt.xlabel("Real (Re)")
plt.ylabel("Imaginary (Im)")
plt.show()
        