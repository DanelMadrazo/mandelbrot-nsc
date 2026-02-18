"""
Mandelbrot Set Generator
Author : [ Danel Madrazo ]
Course : Numerical Scientific Computing 2026
"""
import numpy as np
x = -0.1
y = 0.1

c = x + 1j * y

z = np.zeros(100, dtype=complex)
for n in range(99):
    z[n+1] = z[n]**2 + c
    
    if abs(z[n+1]) > 2:
        print(f"point scapes! iteration number:{n+1}")
        break 
else: 
    print(f"Point didnt scape, iteration number: {n+1}")

            