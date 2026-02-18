"""
Mandelbrot Set Generator
Author : [ Danel Madrazo ]
Course : Numerical Scientific Computing 2026
"""
import numpy as np
import time
import matplotlib as plt

start = time.time()
xmin, xmax, ymin, ymax = -2, 1, -1.5, 1.5
res = 1024

x = np.linspace(xmin, xmax, res)
y = np.linspace(ymin, ymax, res)

def mandelbrot_point(x, y, max_iter = 100):
    c = x + 1j * y
    z = 0j
    for n in range(max_iter):
        z = z**2 + c 
        if abs(z) > 2:
            break    
    return n

iteration_num = np.zeros((res, res))

for i in range(res):
    for j in range (res):
        iteration_num[i, j] = mandelbrot_point(x[j], y[j])
          
elapsed = time.time() - start
print(elapsed)
