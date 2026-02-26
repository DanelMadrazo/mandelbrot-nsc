"""
Mandelbrot Set Generator
Author : [ Danel Madrazo ]
Course : Numerical Scientific Computing 2026
"""
import numpy as np
import time , statistics
import matplotlib.pyplot as plt
#import cProfile , pstats
from numba import njit


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
#L3 MILESTONE 3
@njit
def mandelbrot_naive_numba(xmin, xmax, ymin, ymax, x_res, y_res, max_iter = 100):
    x = np.linspace(xmin, xmax, x_res)
    y = np.linspace(ymin, ymax, y_res)
    
    # OJO: y_res va primero para el número de filas (altura)
    result = np.zeros((y_res, x_res), dtype=np.int32) 
    
    for i in range(y_res):     # OJO: i recorre el eje Y (filas)
        for j in range(x_res): # OJO: j recorre el eje X (columnas)
            c = x[j] + 1j * y[i]
            z = 0j
            n = 0
            while n < max_iter and z.real * z.real + z.imag * z.imag <= 4.0:
                z = z*z + c 
                n += 1
            result[i, j] = n
    return result
@njit
def mandelbrot_point_numba (c , max_iter =100) :
    z = 0j
    for n in range (max_iter):
        if z.real * z.real + z.imag * z.imag >4.0:
            return n
        z = z *z + c
    return max_iter

def mandelbrot_hybrid (xmin, xmax, ymin, ymax, x_res, y_res, max_iter = 100):
    # outer loops still in Python
    x = np.linspace(xmin, xmax, x_res)
    y = np.linspace(ymin, ymax, y_res)

    iteration_num = np.zeros((y_res, x_res))
    
    for i in range(y_res):
        for j in range(x_res):
            c = complex(x[j], y[i])
            n = mandelbrot_point_numba(c, max_iter)
            iteration_num[i, j] = n
    return iteration_num
def bench (fn , * args , runs =5) :
    fn (* args ) # warm -up
    times = []
    for _ in range ( runs ) :
        t0 = time . perf_counter ()
        fn (* args )
        times . append ( time . perf_counter () - t0 )
    return statistics . median ( times )

#L3 MILESTONE 4
@njit
def mandelbrot_numba_typed(xmin, xmax, ymin, ymax, width, height, max_iter =100, dtype = np.float64):
    x = np . linspace ( xmin , xmax , width ). astype ( dtype )
    y = np . linspace ( ymin , ymax , height ) . astype ( dtype )
    result = np . zeros (( height , width ) , dtype = np . int32 )
    for i in range ( height ):
        for j in range ( width ):
            c = x [j] + 1j * y[ i]
            result [i , j ] = mandelbrot_point_numba (c , max_iter )
    return result

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
    #print (f" Median : { median_t:.4f}s "
    #f"( min ={ min( times ):.4f}, max ={ max( times ):.4f})")
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
print(f"Computation with naive took {t_naive:.4f} seconds")

#1024x1024 resolution takes around 4 seconds
#2048x2048 resolution takes around 17 seconds

t_numpy , numpy_result = benchmark(compute_mandelbrot_numpy, C, 100)
print(f"Computation with numpy took {t_numpy:.4f} seconds")


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

#MILESTONE 4
grid_size = [256, 512, 1024, 2048]
xmin, xmax, ymin, ymax = -2, 1, -1.5, 1.5
t = [0] * len(grid_size)
for i in range(len(grid_size)):
    x = np . linspace ( xmin , xmax, grid_size[i]) # 1024 x- values
    y = np . linspace ( ymin , ymax , grid_size[i]) # 1024 y- values
    X , Y = np . meshgrid (x , y) # 2D grids
    C = X + 1j* Y # Complex grid
    t[i], result = benchmark(compute_mandelbrot_numpy, C, 100)
    
plt.figure()
plt.plot(grid_size, t)
plt.title('Problem Size Scaling')
plt.xlabel('grid_scale')
plt.ylabel('time')

#Lecture 3
##MILESTONE 1
# cProfile.run('compute_mandelbrot_naive( xmin, xmax, ymin, ymax, 1024 , 1024)', 'naive_profile.prof')

# cProfile.run('compute_mandelbrot_numpy(C)', 'numpy_profile.prof')
             
# for name in ('naive_profile.prof', 'numpy_profile.prof'):
#     stats = pstats.Stats(name)
#     stats.sort_stats('cumulative')
#     stats.print_stats(10)

_ = mandelbrot_hybrid ( -2 , 1, -1.5 , 1.5 , 64 , 64)
_ = mandelbrot_naive_numba ( -2 , 1, -1.5 , 1.5 , 64 , 64)

#NUMBA (L3 MILESTONE3)
width , height = 1024 , 1024
args = ( -2 , 1, -1.5 , 1.5 , width , height )

t_naive = bench (compute_mandelbrot_naive , * args )
t_numpy = bench (compute_mandelbrot_numpy , C, 100)
t_numba = bench (mandelbrot_naive_numba , * args )

print (f" Naive : { t_naive :.3f}s")
print (f" NumPy : { t_numpy :.3f}s ({ t_naive / t_numpy :.1f}x)")
print (f" Numba : { t_numba :.3f}s ({ t_naive / t_numba :.1f}x)")

#L3 MILESTONE4
for dtype in [np.float32, np.float64]:
    t0 = time . perf_counter ()
    mandelbrot_numba_typed(-2, 1, -1.5, 1.5, 1024, 1024, dtype = dtype)
    print (f"{dtype.__name__}: { time.perf_counter()-t0:.3f}s")
    
    
