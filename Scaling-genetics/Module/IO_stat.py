# -*- coding: utf-8 -*-
"""
Created on 2022/3/1 15:38

@author: shan, gmking

This module is used to save and read parameters of GLC.

1. FRD_protein: dict
    return of count.py > FRD_plot()
    where
    (1) FRD_protein['ab']: tuple (a_Z, b_Z)
            parameters of P(x, b) = a_Z * x^(-b_Z)
    (2) FRD_protein['b_jac']: tuple
            gradient vector used for optimization
    (3) FRD_protein['neg_L']: float
            negative max liklihood. 
            details see Curve_Fitting_MLE.py > L_Zipf_Mandelbrot()
    (4) FRD_protein['length']: int
                total number of proteins in the txt
    (5) FRD_protein['V_1'] = int
                total kinds of proteins in the txt

2. RRD_coordinate: N*2 array, N = number of points
    return of count.py > draw_RRD_plot()
    where
    (1) coordinate[i][0] = x coordinate
    (2) coordinate[i][1] = y coordinate
    
3. Allo_fit: tuple (A, B), (float, float)
    return of allo_chain.py > Allo_plot()
    fitting parameters of Allo(y') = (-A ln y' + B)^2

4. Chain_fit: tuple (A, B, U_Chain), (float, float, int)
    return of allo_chain.py > Chain_plot()
    A, B are fitting parameters of Chain(x') = -A ln x' + B
    U_Chain denotes the the least upper bound of Chain

5. glu: set, {glu_1, glu_2,...,glu_L}
    return of denoise.py > plot_g()
    where glu_k = (lupx, lupy)_k denote the points on scaling line g_k 

6. Rg: (R, ERROR, R_dist, S_value*C_value), a 4-D tuple
    return of denoise.py > rg()
    where
    (1) Rg[0] = R: avg of g_(k+1)/g_k for all k
    (2) Rg[1] = error: standard error for g_(k+1)/g_k
    (3) Rg[2] = R_dist: record g_(k+1)/g_k for every x withour NAN
    (4) Rg[3] = S_value*C_value: SC value, a index to guage the goodness of scaling structure    
    
7. fit_para_best: dict, {'popt', 'pcov', 'score'}
    return of denoise.py > scaling_fit()
    where
    (1) popt, pcov: 2-D list, 2*2-D list
            they come from popt, pcov = curve_fit()
            fitting parameters of scaling function for the best basis
            popt (parameters optimization) contains q and C of fun_theory(x, q, C)
            pcov (parameters covariance) contains the covariance
            see innner function > fun_theory()

            for curve_fit(),
            see scipy.optimize.curve_fit for data structure
            #url = https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    (2) score: float
            The goodness of fitting
            see Sec. X in SI for detals


8. degree_component: dict, 
    return of network.py > plot_degree_compo()
    where
    (1) degree_component['abc']: tuple (a_ZM, b_ZM, c_ZM)
            parameters of P(x, b, c) = a_ZM*(x + c_ZM)^-b_ZM
    (2) degree_component['bc_jac']: tuple
            gradient vector used for optimization
    (3) degree_component['neg_L']: float
            negative max liklihood. 
            details see Curve_Fitting_MLE.py > L_Zipf_Mandelbrot()


"""

from ast import literal_eval
import re

check_list = ['FRD_protein', 'RRD_coordinate', 'Allo_fit', 'Chain_fit', 'glu', 'Rg', 'fit_para_best', 'degree_component']

def save_parameters(para_filename, data_set, Path = ''):
    if Path == '':
        f = open(para_filename, 'w', encoding = 'utf-8')
    else:
        f = open(Path + para_filename, 'w', encoding = 'utf-8')
    
    for d in data_set:
        if d not in check_list:
                print('unknown parameter: %s' % d)
        else:
              print(d)
        f.write('#' + d + '\n')
        f.write(str(data_set[d]))
        f.write('\n')
    f.close()


def read_parameters(para_filename, Path = ''):
    data_set = {}
    if Path == '':
        f = open(para_filename, 'r', encoding = 'utf-8')
    else:
        f = open(Path + para_filename, 'r', encoding = 'utf-8')
    
    for i in f:
        if '#' in i:
            label_name = re.split('#|\n', i)[1]
            if label_name not in check_list:
                print('unknown parameter: %s' % label_name)
        else:
            data_set[label_name] = literal_eval(i)
            print(type(data_set[label_name]))
    f.close()
    return data_set
