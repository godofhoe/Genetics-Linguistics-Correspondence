import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import zeta #https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.special.zeta.html
'''
If you need detail explaination, please look Fitting_MLE.ipynb
'''

def incomplete_harmonic(x):
    x_min = x[0]
    x_max = x[1]
    s = x[2]
    P = 0
    for k in range(int(x_min) , int(x_max) + 1):
        P = P + 1 / k ** s
    return P

def incomplete_shifted_harmonic(x, a):
    x_min = x[0]
    x_max = x[1]
    s = x[2]
    P = 0
    for k in range(int(x_min) , int(x_max) + 1):
        P = P + 1 / (k + a) ** s
    return P

def Zipf_law(x, s, C):
    return C * x ** (-s)

def Zipf_Mandelbrot(x, s, C, a):
    return C * (x + a) ** (-s)

def Two_to_One(y):
    #y = ([rank], [frequency of the rank])
    Y = []
    for i in y[0]:
        Y.append(i)
    for i in y[1]:
        Y.append(i)
    return Y

def One_to_Two(Y):
    y = [[], []]
    length = len(Y) * 0.5
    for i in range(int(length)):
        y[0].append(Y[i])
    for i in range(int(length)):
        y[1].append(Y[i + int(length)])
    return y

def L_Zipf(s, Y):
    y = One_to_Two(Y)
    #y = ([rank], [frequency of the rank])
    ln = 0
    for i in range(len(y[1])):
        ln = ln + y[1][i] * np.log(y[0][i])
    N = sum(y[1])
    x = (int(min(y[0])), int(max(y[0])), s) #y[2] is exponent
    return s * ln + N * np.log(incomplete_harmonic(x))

def L_Zipf_Mandelbrot(t, Y):
    s = t[0]
    a = t[1]
    y = One_to_Two(Y)
    #y = ([rank], [frequency of the rank])
    ln = 0
    for i in range(len(y[1])):
        ln = ln + y[1][i] * np.log(y[0][i] + a)
    y = One_to_Two(Y)
    N = sum(y[1])
    x = (int(min(y[0])), int(max(y[0])), s) #y[2] is exponent
    return s * ln + N * np.log(incomplete_shifted_harmonic(x, a))

def L_Zipf_zeta(s, Y):
    y = One_to_Two(Y)
    #y = ([rank], [frequency of the rank])
    ln = 0
    for i in range(len(y[1])):
        ln = ln + y[1][i] * np.log(y[0][i])
    N = sum(y[1])
    return s * ln + N * np.log(zeta(s, int(min(y[0]))))