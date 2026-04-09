# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:18:04 2026
Author : [ Danel Madrazo ]
Course : Numerical Scientific Computing 2026
"""
import numpy as np
def find_machine_epsilon(dtype=np.float64):
    eps = dtype(1.0)
    while dtype(1.0) + eps / dtype(2.0) != dtype(1.0):
        eps = eps / dtype(2.0)
    return eps
def quadratic_naive(a, b, c):
    t = type(a)
    disc = t(np.sqrt(b*b - t(4)*a*c))
    x1 = (-b + disc) / (t(2)*a)
    x2 = (-b - disc) / (t(2)*a)
    return x1, x2

# for dtype in [np.float16, np.float32, np.float64]:
#     computed = find_machine_epsilon(dtype)
#     reference = np.finfo(dtype).eps
#     print(f'{dtype.__name__}:')
#     print(f" Computed: {float(computed):.4e}")
#     print(f" np.finfo: {float(reference):.4e}")
#     print()

for dtype in [np.float32, np.float64]:
    a, b, c = dtype(1.0), dtype(-10000.0001), dtype(1.0)
    x1, x2 = quadratic_naive(a, b, c)
    print(f'{dtype.__name__}: x1 = {float(x1):.4f}, x2 = {float(x2):.10f}')
    