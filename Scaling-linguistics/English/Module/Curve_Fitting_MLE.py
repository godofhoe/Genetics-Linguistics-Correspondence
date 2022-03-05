import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import zeta #https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.special.zeta.html
'''
If you need theoretical explaination, please look Fitting_MLE.ipynb
'''

def incomplete_harmonic(x):
    x_min = x[0]
    x_max = x[1]
    b = x[2]
    P = 0
    for k in range(int(x_min) , int(x_max) + 1):
        P = P + 1 / k ** b
    return P

def incomplete_shifted_harmonic(x, c):
    x_min = x[0]
    x_max = x[1]
    b = x[2]
    P = 0
    for k in range(int(x_min) , int(x_max) + 1):
        P = P + 1 / (k + c) ** b
    return P

def Zipf_law(x, a, b):
    return a * x ** (-b)

def Zipf_Mandelbrot(x, a, b, c):
    return a * (x + c) ** (-b)

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

def L_Zipf(b, Y):
    #negative log likelihood of Zipf
    y = One_to_Two(Y)
    #y = ([rank], [frequency of the rank])
    ln = 0
    for i in range(len(y[1])):
        ln = ln + y[1][i] * np.log(y[0][i])
    N = sum(y[1])
    x = (int(min(y[0])), int(max(y[0])), b) #b is exponent
    return b * ln + N * np.log(incomplete_harmonic(x))

def L_Zipf_Mandelbrot(bc, Y):
    # "negative" log likelihood of Zipf_Mandelbrot
    b = bc[0]
    c = bc[1]
    y = One_to_Two(Y)
    #y = ([rank], [frequency of the rank])
    ln = 0
    for i in range(len(y[1])):
        ln = ln + y[1][i] * np.log(y[0][i] + c)
    y = One_to_Two(Y)
    N = sum(y[1])
    x = (int(min(y[0])), int(max(y[0])), b) #b is exponent
    return b * ln + N * np.log(incomplete_shifted_harmonic(x, c))

def L_Zipf_zeta(b, Y):
    # "negative" log likelihood of Zipf normalizes by zeta function
    y = One_to_Two(Y)
    #y = ([rank], [frequency of the rank])
    ln = 0
    for i in range(len(y[1])):
        ln = ln + y[1][i] * np.log(y[0][i])
    N = sum(y[1])
    return b * ln + N * np.log(zeta(b, int(min(y[0]))))

def AICc_choose(L, k, N):
    '''choose the best model by comparing their AICc
    
    L=(L1, L2, L3, ...) "negative" log likelihood of model 1, model 2, model 3...
    k=(k1, k2, k3, ...) number of parameters used in model 1, model 2, model 3...
    N is sample size
    AICc = 2k+2L + 2k*(k+1)/(N-k-1)
    
    the best model = min AICc
    '''
    aicc = [2*(k[n] + L[n]) + 2*k[n]*(k[n] + 1)/(N - k[n] - 1) for n in range(len(L))]
    min_aicc_index = aicc.index(min(aicc))
    return min_aicc_index, aicc

def AIC_choose(L, k):
    '''choose the best model by comparing their AIC
    when sample size N is large enough, AICc will approach to AIC
    
    L=(L1, L2, L3, ...) "negative" log likelihood of model 1, model 2, model 3...
    k=(k1, k2, k3, ...) number of parameters used in model 1, model 2, model 3...
    AIC = 2k+2L 
    
    the best model = min AIC
    '''
    aic = [2*(k[n] + L[n]) for n in range(len(L))]
    min_aic_index = aic.index(min(aic))
    return min_aic_index, aic

def Z_ZM_choose(Y):
    '''
    choose Zipf or Zipf-Mandelbrot to be the dest fits depend on aicc
    '''
    y = One_to_Two(Y)
    #Estimate exponent. This action can make reduce the error of initial value guess.
    freq_M, freq_m = max(y[1]), min(y[1])
    rank_M, rank_m = max(y[0]), min(y[0])
    b_0 = np.log(freq_M / freq_m) / np.log(rank_M / rank_m)
    
    #fit Zipf: P(x, b)=a_Z/x^b_Z
    res_Z = minimize(L_Zipf, b_0, Y)
    b_Z = res_Z['x']
    t_Z = (int(min(y[0])), int(max(y[0])), b_Z)
    a_Z = 1 / incomplete_harmonic(t_Z)
    Z_para = (a_Z, b_Z)
    
    #fit Zipf-Mandelbrot: P(x, b, c)=a_ZM/(x+c_ZM)^b_ZM
    res_ZM = minimize(L_Zipf_Mandelbrot, (b_0, 0), Y)
    b_ZM = res_ZM['x'][0]
    c_ZM = res_ZM['x'][1]
    t_ZM = [int(min(y[0])), int(max(y[0])), b_ZM]
    a_ZM = 1 / incomplete_shifted_harmonic(t_ZM, c_ZM)
    ZM_para = (a_ZM, b_ZM, c_ZM)
    
    #comparing their aicc
    L = [res_Z.fun, res_ZM.fun]
    k = [1,2]
    N = sum(y[1])
    aicc_index, aicc = AICc_choose(L, k, N)
    
    model_list = ['Zipf', 'Zipf-Mandelbrot']    
    print('the best fits is model %s' % model_list[aicc_index])
    
    return aicc_index, aicc, Z_para, ZM_para