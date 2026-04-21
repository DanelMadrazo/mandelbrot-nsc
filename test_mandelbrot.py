# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 13:38:03 2026
Author : [ Danel Madrazo ]
Course : Numerical Scientific Computing 2026
"""

import pytest
from mandelbrot import mandelbrot_point

#TEST 1
def test_origin_dont_escape():
    result = mandelbrot_point(0 + 0j, max_iter=100)
    assert result == 100
    
#TEST 2
def distant_point_escape():
    result = mandelbrot_point(5 + 0j, max_iter = 100)
    assert result < 100
    
#TEST 3
Known_cases = [
    (0 + 0j, 100, 100),     #Origin (in)
    (5.0 + 0j, 100, 0),     #Distant (Out)
    (-2.5 + 0j, 100, 0),    #Left point (Out)
    (0.25 + 0j, 100, 100),  #Right peak (In)    
    ]

@pytest.mark.parametrize('c, max_iter, expected_result', Known_cases)
def test_parametrized_cases(c, max_iter, expected_result):
    assert mandelbrot_point(c, max_iter) == expected_result
    


