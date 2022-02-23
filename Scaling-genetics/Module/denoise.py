# -*- coding: utf-8 -*-
'''
@author  gmking

This module is used to run all file in the document "input" once instead of running case by case.
The functions here won't show plots. 
If you want to see every picture of your txt, you should use Run_case_by_case.ipynb and DONOT import this module.
'''

import random 
import bisect 
import math 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .count import *
from .Curve_Fitting_MLE import *
from scipy.optimize import curve_fit

def chose_point(m, n, V, H, big, longest):
    '''chose the points in (m, n), (m+1, n+1), ... blocks, where m > n.
    return: points = [[(x_m,y_n),...], [(x_(m+1), y_(n+1)),...], ...]
    
    the position of (m, n) block is the same as element (m*n) in matrix (row m and column n)
    
    '''
    points = [[] for j in range(min(len(V) - n, len(H) - m))]
    for j in range(min(len(V) - n, len(H) - m)):
        #https://thispointer.com/python-pandas-select-rows-in-dataframe-by-conditions-on-multiple-columns/
        V_protein = big.loc[(big['proteinRank'] <= V[j+n-1]) & (big['proteinRank'] > V[j+n])]
        for i in range(longest): #draw 0th_dom ~ (longest-1)_th_dom
            H_dom = V_protein[(V_protein[str(i) + "th_dom_rank"] <= H[j+m-1]) & (V_protein[str(i) + "th_dom_rank"] > H[j+m])]
            ind = H_dom.index #if I don't use index here, the content of points will be series not value
            for k in ind:
                points[j].append((H_dom.loc[k, 'proteinRank'], H_dom.loc[k, str(i) + "th_dom_rank"]))
    return points

def sort_point(m, n, p):
    '''
    sort points in each block to make analysis easier
    '''
    for i in p:
        #sort list with key
        i.sort(key = lambda x: x[1], reverse = True)
    px = []
    py = []
    for i in range(5):
        if p[i] == []:
            print ('the (%d, %d) block have no points.' % (i+m, i+n))
        for j in p[i]:
            px.append(j[0])
            py.append(j[1])
    return px, py

def left_upper(m, n, V, H, big, longest):
    points = chose_point(m, n, V, H, big, longest)
    lup = [[] for i in range(len(points))]
    for i in range(len(points)):
        for p in points[i]:
            if p[1] > p[0]*(H[i+m-1] - H[i+m])/(V[i+n-1] - V[i+n]) + H[i+m-1] - V[i+n-1]*(H[i+m-1] - H[i+m])/(V[i+n-1] - V[i+n]):
                lup[i].append(p)
    return lup

def Tv(r, p):
    '''
    input:
    r: ndarray
    data points after denosing, where r = [rx1, rx2, ..., rxn, ry1, ry2, ..., ryn]
    p: ndarray
    data with noise, where p = [x1, x2, ..., xn, y1, y2, ..., yn]
    
    output:
    a function used to denoise
    '''
    def penalty(u, M):
        #This penalty function is used to lower the influence of outliers
        if abs(u) <= M:
            return u ** 2
        elif abs(u) > M:
            return M * (2 * abs(u) - M)        
    t = 0
    for i in range(len(p)):
        t = t + penalty(r[i] - p[i], 50)
    
    #see taxicab distance
    Lambda = 1 #regularziation parameters
    rl = int(len(r)/2)
    return sum(np.square(r[1:rl]-r[:rl - 1])) + sum(np.square(r[rl + 1:]-r[rl:-1])) + Lambda*t

def DENOISE(m, n, V, H, big, longest, toler):  
    lup = left_upper(m, n, V, H, big, longest)
    lupx, lupy = sort_point(m, n, lup)
    lupxy = np.array(lupx + lupy)  #lupxy = [x1, x2, ..., xn, y1, y2, ..., yn]
    RR = minimize(Tv, lupxy, args = lupxy, method='CG', tol = toler)
    luptx = RR.x[:int(len(RR.x)/2)]
    lupty = RR.x[int(len(RR.x)/2):]
    return luptx, lupty

def moving_avg(px, py, Range, N = 50):
    '''use moving average on (px,py), ruturn p_avg
    ---input:
    px, py: 1-D list
    Range: the x-range of data points
    N: number of segmentation, i.e. segment pi into N block. the moving period = (max(Range)-min(Range))/N
    
    ---return:
    p_avg: (px,py) after moving average
    '''
    period = (max(Range) - min(Range))/N
    px_avg = [[] for i in range(N)]
    py_avg = [[] for i in range(N)]
    p = [(px[i], py[i]) for i in range(len(px))]
    
    for j in p:
        #check every points
        for i in range(N):
            if (j[0] >= min(Range) + i*period) & (j[0] <= min(Range) + (i+1)*period):
                px_avg[i].append(j[0])
                py_avg[i].append(j[1])
                break    
    x_avg, y_avg = [], []
    for i in range(N):
        if (px_avg[i] == []) or (py_avg[i] == []):
            x_avg.append(float('nan'))
            y_avg.append(float('nan'))
        else:
            x_avg.append(np.mean(px_avg[i]))
            y_avg.append(np.mean(py_avg[i]))        
    return x_avg, y_avg


def plot_g(n, V, H, big, name, longest, toler = 50, avg_N = 50):
    '''
    ---input
    n: integer, this function will select points on scaling line from g_1 to g_n
    toler: float, tolerance. Increase this value will speed up the minimization process but decline in performance.
    avg_N: number of segmentation (N block) that used to  doing moving avgerage.
    
    ---return
    g: set, {g_1, g_2,...,g_n}, where g_k = (x_avg, y_avg), x/y_avg denote points after moving average
    glu: set, {glu_1, glu_2,...,glu_n}, where glu_k = (lupx, lupy)_k, lupx/y is points on scaling line
    '''
    #-----------------------------plot horizontal and vertical lines
    Slice_number = 50 #this value decide the number of points on horizontal and vertical lines
    number_of_lines = 4
    x_range = np.linspace(0, max(V), Slice_number)
    y_range = np.linspace(0, max(H), Slice_number)
        
    for i in range(number_of_lines):
        x_const = [V[i] for j in range(Slice_number)]#x_const =[V[i], V[i], ..., V[i]], Slice_number elements
        y_const = [H[i] for j in range(Slice_number)] #y_const =[H[i], H[i], ..., H[i]], Slice_number elements
        plt.plot(x_range, y_const) #plot y=H[i]
        plt.plot(x_const, y_range) #plot x=V[i]   
    #-----------------------------   
    g = {}
    glu = {}
    #plt.locator_params(axis='y', nbins=5)
    #pick up points on scaling line
    for M in range(1, n+1):
        (M, N) = (M, 1)
        points = chose_point(M, N, V, H, big, longest)
        px, py = sort_point(M, N, points)
        plt.plot(px, py,'o', markersize = '4')
        if M == 1:
            luptx, lupty = px, py
        else:
            luptx, lupty = DENOISE(M, N, V, H, big, longest, toler)
        glu['g' + str(M)] = (luptx, lupty)
        plt.plot(luptx, lupty,'.' ,markersize = '4', color = '#e9bf53')
        #moving average
        Range = [0.25*V[0], V[0]]
        x_avg, y_avg = moving_avg(luptx, lupty, Range, avg_N)
        g['g' + str(M)] = (x_avg, y_avg)
    plt.xlim([0,V[0]*1.03])
    plt.ylim([0,H[0]*1.03])
    plt.xlabel('protein', size = 15)
    plt.ylabel('domain', size = 15)  
    plt.title(name, size = 20)
    plt.show()
    return g, glu

def rg(name, g, FORMAT, Path = ''):
    '''plot r_g of your data
    
    ------paras
    name: str
       name of your r_g plot
    g: set, contain points on g1 ~ gn after moving average
       output of plot_g(n, V, H) #suggestion: use g = glu(non-average data), not g = g(average data)
    FORMAT: string
       The format of your RRD plot. Most backends support png, pdf, ps, eps and svg. 
       else: just show plot instead of saving.
    
    Path: file path for saving picture
       Default: save at current document
       
    ------output
    R: avg of g_(n+1)/g_n for all n
    error: standard error for g_(n+1)/g_n
    figure including information about R and error
    '''
    avg_N = max([len(g[i][0]) for i in g])
    lg = len(g)
    y = {}
    x = {}
    STD = {} #STD of g_n/g_(n+1)
    weight = {} #number of data of g_n/g_(n+1)
    r = {} #average ratio of g_n/g_(n+1)
    for n in range(1, lg):
        gn1 = 'g' + str(n+1)
        gn = 'g' + str(n)
        y[gn1 + '/' + gn] = [g[gn1][1][j]/g[gn][1][j] for j in range(avg_N)]
        x[gn1 + '/' + gn] = [0.5*g[gn1][0][j] + 0.5*g[gn][0][j] for j in range(avg_N)]
        
        y_n = [i for i in y[gn1 + '/' + gn] if i==i] # this is y without NAN
        STD[gn1 + '/' + gn] = round(np.std(y_n), 3)
        weight[gn1 + '/' + gn] = len(y_n)
        r[gn1 + '/' + gn] = round(np.mean(y_n), 3)
    
    fig, ax = plt.subplots()
    #calculate SC value excluding g2/g1
    error = {}
    for i in STD:
        if i != 'g2/g1':
            error[i] = STD[i]
    del weight['g2/g1'], r['g2/g1']
    tot = sum([weight[w] for w in weight])
    R = sum([weight[i]*r[i]/tot for i in weight])
    ERROR = (sum([weight[i]*error[i]**2/tot for i in weight]))**0.5
    EE = (sum([weight[i]*error[i]/tot for i in weight]))
    
    
    C = {}
    for i in x:
        px = x[i]
        py = y[i]
        std = STD[i]
        if i != 'g2/g1':
            C[i] = weight[i]/len(py)
            if C[i] < 0.8:
                print('C < 0.8: %s, %f' % (i, C[i]))
        ax.errorbar(px, py, yerr = std) #plot errorbar
        plt.plot(px, py,'o', markersize = '4', label = i)
        plt.legend(loc = 'best', prop = {'size': 15})
    
    C_value = np.mean([C[i] for i in C])
    S_value = 1 - ERROR/R
    
    xmin, xmax = plt.xlim([0,None])
    ymin, ymax = plt.ylim([0,None])
    plt.text(xmax*0.98, ymax*0.5, '$r_g=%.3f\pm %.3f$' % (R,ERROR), fontsize=35, verticalalignment='bottom', horizontalalignment='right')
    plt.text(xmax*0.9, ymax*0.35, '$S=%.3f$' % (S_value), fontsize=35, verticalalignment='bottom', horizontalalignment='right')
    plt.text(xmax*0.9, ymax*0.2, '$C=%.3f$' % (C_value), fontsize=35, verticalalignment='bottom', horizontalalignment='right')
    plt.text(xmax*0.9, ymax*0.05, '$SC=%.3f$' % (S_value*C_value), fontsize=35, verticalalignment='bottom', horizontalalignment='right')
    plt.xlabel('$x$', size = 20)
    plt.ylabel('$r_g(x)$', size = 20)
    ax.tick_params(axis='x', labelsize=15) 
    ax.tick_params(axis='y', labelsize=15)
    #https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.ticklabel_format.html
    #https://atmamani.github.io/cheatsheets/matplotlib/matplotlib_2/
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-1,1)) 
    ax.xaxis.set_major_formatter(formatter) 
    #https://stackoverflow.com/questions/34227595/how-to-change-font-size-of-the-scientific-notation-in-matplotlib
    ax.yaxis.offsetText.set_fontsize(15)
    ax.xaxis.offsetText.set_fontsize(15)
    
    #https://stackoverflow.com/questions/6774086/why-is-my-xlabel-cut-off-in-my-matplotlib-plot
    plt.gcf().subplots_adjust(left = 0.17, bottom = 0.17)
    plt.title(name, size = 20)
    try:
        if Path == '':
            fig.savefig('rg of ' + name + '.' + FORMAT, dpi = 400, format = FORMAT)
            plt.show()
        else:
            fig.savefig(Path + 'rg of ' + name + '.' + FORMAT, dpi = 400, format = FORMAT)
            plt.close()
    except:
        plt.show()
    return (R, ERROR)

def scaling_fit(data, Rg, V, H, Zipf, name, FORMAT = 'pdf', Path = ''):
    '''find out best fitting curve for scaling lines
    use fun to be fitting model, select g1~gn to be basis of scaling function, after that find out the best basis and parameters
    by check deviation of different basis.
    ### Notice: we don't use g1 as basis, and exclude g1 when calculate deviation
    For instance, use g2 to be basis
    0. fitting g2 with fun
    1. g3 = Rg*g2, g4 = Rg^2*g2......
    2. tot_Dev['g2'] = sum((y_i - g_i)**2), where y_i is real data and g_i is fitting data
    3. check all possible and resonable basis, findout the smallest tot_Dev['gn'] 
    4. calculate fitting score, base on an empirical truth that good fitting use less data
    ------paras
    data: set of points on scaling line, it is output of 
          g, glu = plot_g(4, V, H, big, longest, toler = 50, avg_N = 50)
          these data should contain no nan. I don't suggest use moving average data
    Rg: ratio of scaling curve
        Instead of Rg, you can try RH
        Rg = rg(name, SP, FORMAT, g)
        where Rg[0] = mean, Rg[1] = std
        RH = which_plot(x, SP, FORMAT, max_range, shift)
        where RH[0] = mean, RH[1] = std, RH[2] = shift
    Zipf: tuple (c, s), where C is the leading coffecient and s is the exponent of frequency-rank distribution of protein        
    ------output
    1. a picture with best fitting curve
    2. fitting parameters
    
    '''
    fig, ax = plt.subplots()
    #plt.locator_params(axis='y', nbins=5)
    #-----------------------------plot horizontal and vertical lines
    Slice_number = 50 #this value decide the number of points on horizontal and vertical lines
    number_of_lines = 4
    x_range = np.linspace(0, max(V), Slice_number)
    y_range = np.linspace(0, max(H), Slice_number)
        
        
    for i in range(number_of_lines):
        x_const = [V[i] for j in range(Slice_number)] #x_const =[V[i], V[i], ..., V[i]], Slice_number elements
        y_const = [H[i] for j in range(Slice_number)] #y_const =[H[i], H[i], ..., H[i]], Slice_number elements
        plt.plot(x_range, y_const) #plot y=H[i]
        plt.plot(x_const, y_range) #plot x=V[i]   
    #-----------------------------  
    def fun_(x, q, s, t):
        '''theory of scaling curve                
        '''
        return q*Rf**(Zipf[0]*(x)**-(Zipf[1]) - s*(x)**-t)
    
    def fun(x, q, s):
        return q*x**s
    number = len(data) #number of scaling lines need fitting
    q0 = (100, 0.5) #initial guess
    fit_para = {}
    tot_Dev = {}
    for gn in data:
        if gn != 'g1':           
            theo = {}
            Dev = {}
            #popt is the optimal values for the parameters (q, s, t)
            popt, pcov = curve_fit(fun, data[gn][0], data[gn][1], q0, bounds =(0, [np.inf, 1.5]))
            fit_para[gn] = (popt, pcov)   
            n = int(gn.split('g')[1]) #ex: gn = 'g2' then n = 2
            for N in range(2, number + 1):
                GN = 'g' + str(N)
                theo[GN] = fun(data[GN][0], *popt) * Rg**(N - n) #ex: n=2, GN=4 then theo['g3'] = fun*Rg^2
                diff = theo[GN] - data[GN][1]
                Dev[GN] = np.sum(np.square(diff))
            tot_Dev[gn] = np.sum([Dev[i] for i in Dev])
        
    best = min(tot_Dev, key = tot_Dev.get) #Get the key corresponding to the minimum value within a dictionary
    n = int(best.split('g')[1]) #ex: best = 'g2' then n = 2
    
    for N in range(1, number + 1):
        GN = 'g' + str(N)
        theo[GN] = fun(data[GN][0], *fit_para[best][0]) * Rg**(N - n) #ex: n=2, N_G=1 then theo['g1] = Rg^(-1)* fun
        plt.plot(data[GN][0], data[GN][1], '.', markersize = '4', color ='#e9bf53')
        plt.plot(data[GN][0], theo[GN], 'o', markersize = '4')
    xm, xM = plt.xlim([0,V[0]*1.03])
    ym, yM = plt.ylim([0,H[0]*1.03])
    
    #-------------------------------------
    A = format(popt[0], '#.4g')  # give 4 significant digits
    if A[-1] == '.':
        A = A[:-1]
    B = format(popt[1], '#.4g')  # give 4 significant digits
    if B[-1] == '.':
        B = B[:-1]
    #plt.text(0.05*xM, 0.05*yM, r'$f_%d=%s \times %.3f^{%s x^{-%.2f}}$' % (n, A, Rf, B, popt[2]), fontsize=24, color ='black')
    #-------------------------------------
    
    
    plt.xlabel('protein', size = 15)
    plt.ylabel('domain', size = 15)  
    plt.title(name, size = 20)
    
    #the following part is used to calculate fitting score, base on an empirical truth that good fitting use less data
    dx_min = {i:min(data[i][0]) for i in data} #find minima x in data
    if (V[0] - dx_min[min(dx_min)]) < 0.75*V[0]:
        score = 1
    elif V[0] > (V[0] - dx_min[min(dx_min)]) >= 0.75*V[0]:
        score = (0.75*V[0])/(V[0] - dx_min[min(dx_min)])
    else:
        score = 0.5    
    
    plt.text(0.05*xM, 0.16*yM, 'fitting score: %.3f' % score, fontsize=24, color ='black')
       
    try:
        if Path == '':
            fig.savefig('fitting ' + name + '.' + FORMAT, dpi = 300, format = FORMAT)
            plt.show()
        else:
            fig.savefig(Path + 'fitting ' + name + '.' + FORMAT, dpi = 300, format = FORMAT)
            plt.close()
    except:
        plt.show()
    return fit_para[best]

def fit_with_cut(data, Rg, V, H, Zipf, name, FORMAT, Path = ''):
    '''
    fit data bigger than 0.25*V[0] to rise accuracy of fitting
    if 0.25*V[0] is not small enough, lowering the low bound of data automatically
    
    ------paras
    data: set of points on scaling line, it is output of 
          g, glu = plot_g(4, V, H, big, longest, toler = 50, avg_N = 50)
          these data should contain no nan. I don't suggest use moving average data
    Rg: ratio of scaling curve
        Instead of Rg, you can try RH
        Rg = rg(name, SP, FORMAT, g)
        where Rg[0] = mean, Rg[1] = std
        RH = which_plot(x, SP, FORMAT, max_range, shift)
        where RH[0] = mean, RH[1] = std, RH[2] = shift
    Zipf: tuple (c, s), where C is the leading coffecient and s is the exponent of frequency-rank distribution of protein        
    ------output
    1. an fitted plot that has fitting score
    2. fit_para: fitting parameters for scaling function
    '''
    data_range = [0.25 - i*0.01 for i in range(26)]
    check = 0
    for dr in data_range:
        try:
            D = {}
            for gn in data:
                b = [[],[]]                
                for i in range(len(data[gn][0])):        
                    if data[gn][0][i] >= dr*V[0]:
                        b[0].append(data[gn][0][i])
                        b[1].append(data[gn][1][i])
                D[gn] = (b[0], b[1])
            fit_para = scaling_fit(D, Rg, V, H, Zipf, name, FORMAT, Path)
            print('fitting range = [%d, %d]' % (dr*V[0], V[0]))
            check = 1
            return fit_para
            break
        except RuntimeError:
            pass
    if check == 0 :
        print('Can not find best parameters in data range.')