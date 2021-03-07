import numpy as np
from math import sqrt

lagrange_test_rec = [[0,0], [1,0], [0,1], [1,1]]
rec_coefficients = []

rec_coefficients.append([1,-1,-1,1])
rec_coefficients.append([0,1,0,-1])
rec_coefficients.append([0,0,1,-1])
rec_coefficients.append([0,0,0,1])


# Define the shape functions on arbitrary elements
def shape_rec(x, y, n):
    return rec_coefficients[n][0] + rec_coefficients[n][1] * x + rec_coefficients[n][2] * y + rec_coefficients[n][3] * x * y


def grad_rec(x, y, n):
    gradient = np.array([rec_coefficients[n][1] + rec_coefficients[n][3] * y,
                         rec_coefficients[n][2] + rec_coefficients[n][3] * x])
    return gradient

lagrange_test_tri = [[0,0], [1,0], [0,1]]
tri_coefficients = []

tri_coefficients.append([1,-1,-1])
tri_coefficients.append([0,1,0])
tri_coefficients.append([0,0,1])

# Define the shape functions on arbitrary elements
def shape_tri(x, y, n):
    return tri_coefficients[n][0] + tri_coefficients[n][1] * x + tri_coefficients[n][2] * y


def grad_tri(x, y, n):
    gradient = np.array([tri_coefficients[n][1], tri_coefficients[n][2]])
    return gradient


