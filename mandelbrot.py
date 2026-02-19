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

#L2 MILESTONE 2
def compute_mandelbrot_numpy(C, max_iter = 100):
    Z = np.zeros_like(C)
    M = np.zeros(C.shape, dtype=int)
    for i in range(max_iter):  
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask]**2 + C[mask]
        M[mask] += 1
    return M

#L2 MILESTONE 3 
def row_sums(N,A):
    for i in range(N): s = np.sum(A[i, :])
    return s
def column_sums(N,A):
    for j in range(N): s = np.sum(A[:, j])
    return s   

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

xmin, xmax, ymin, ymax = -2, 1, -1.5, 1.5
res_x, res_y = 1024, 1024

x = np . linspace ( xmin , xmax, res_x) # 1024 x- values
y = np . linspace ( ymin , ymax , res_y) # 1024 y- values
X , Y = np . meshgrid (x , y) # 2D grids
C = X + 1j* Y # Complex grid
print (f" Shape : {C. shape }") # (1024 , 1024)
print (f" Type : {C. dtype }") # complex128

#STEP 4
t_naive, naive_result  = benchmark(compute_mandelbrot_naive, xmin, xmax, ymin, ymax, res_x, res_y, 100)


#1024x1024 resolution takes around 4 seconds
#2048x2048 resolution takes around 17 seconds

print(f"Computation with naive took {t_naive:.3f} seconds")

t_numpy , numpy_result = benchmark(compute_mandelbrot_numpy, C, 100)


if np.allclose ( naive_result , numpy_result ):
    print (" Results match !")
else :
    print (" Results differ !")
# Check where they differ :
diff = np .abs ( naive_result - numpy_result )
print (f" Max difference : { diff . max ()}")
print (f" Different pixels : {( diff > 0). sum ()}")


#STEP 5
plt.figure()
plt.imshow(numpy_result, cmap = 'viridis')
plt.title("Mandelbrot")
plt.xlabel("Real (Re)")
plt.ylabel("Imaginary (Im)")
plt.show()

#MILESTONE 3
N = 10000
A = np.random.rand(N, N)
t_raw, s_raw = benchmark(row_sums, N, A)
t_column, s_column = benchmark(column_sums, N, A)
print(f"sum of raws took {t_raw:.4f} seconds")
print(f"sum of columns took {t_column:.4f} seconds")

#With 'asfortranarray' the raw which was way faster the previous time is now slower
A_f = np.asfortranarray(A)
t_raw, s_raw = benchmark(row_sums, N, A_f)
t_column, s_column = benchmark(column_sums, N, A_f)
print(f"sum of raws took {t_raw:.4f} seconds with 'asfortanarray' ")
print(f"sum of columns took {t_column:.4f} seconds 'asfortanarray' ")

