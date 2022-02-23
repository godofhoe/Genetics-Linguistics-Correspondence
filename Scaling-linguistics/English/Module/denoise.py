# -*- coding: utf-8 -*-
'''
@author  gmking

This module is used to fullfil the techniques mentioned in Sec. V and appendix of SI. 

The main quests are 
1. de-nosie and find the upper envelope
2. calculate SC value
3. fitting scaling lines with the approximated analytic form

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

def choose_point(m, n, V, H, big, longest):
    '''chose the points in {m, n}, {m+1, n+1}, ... rectangles, where m <= n.
    {m, n} rectangle denotes those points whose x in (V_m+1, V_m] while y in (H_n+1, H_n]
    See Eq. (3) in SI
    
    ---parameters
    1. m, n: int
       m, n >= 1
    2. V, H: list or np.array
       V and H are the coordinates of the sequence {V} and {H}.
       You should get these two from 
             V, H = geometric_sequence(word, syl)
       where geometric_sequence(word, syl) is the function of count.py

    3. big, longest: pandas.DataFrame, int
        the output of the function info()    
    
    ---return
    points = [[(x_m,y_n),...], [(x_(m+1), y_(n+1)),...], ...]
    
    ps: in each {m, n} = [(x_m,y_n),...], the points are sorted according to their x values (big to small)
    '''
    num_rect = min(len(V) - m, len(H) - n) #total number of chosen rectangle
    points = [[] for i in range(num_rect)]
    for i in range(num_rect):
        #https://thispointer.com/python-pandas-select-rows-in-dataframe-by-conditions-on-multiple-columns/
        V_word = big.loc[(big['wordRank'] <= V[i+m-1]) & (big['wordRank'] > V[i+m])]
        for j in range(longest): #0th_syl ~ (longest-1)_th_syl
            H_syl = V_word[(V_word[str(j) + "th_syl_rank"] <= H[i+n-1]) & (V_word[str(j) + "th_syl_rank"] > H[i+n])]
            ind = H_syl.index #if I don't use index here, the content of points will be series not value
            for k in ind:
                points[i].append((H_syl.loc[k, 'wordRank'], H_syl.loc[k, str(j) + "th_syl_rank"]))
    
    #sort points in each rectangle to make analysis easier and improve the performance of our algorithm
    for i in points:
        #sort list with key
        i.sort(key = lambda x: x[0], reverse = True)
    
    return points

def sep_point(m, n, points):
    '''To plot, we need to separate x and y value of choose_point()
    In order words, the list [[(x_1,y_1),...]_(m,n), [(x_1,y_1),...]_(m+1,n+1), ...] will become
    px = [x_m, ..., x_(m+1), ...] and py = [y_n, ..., y_(n+1), ...] after using sep_point
    
    ---parameters
    1. m, n: int
        the same as m, n in choose_point()
    2. points: array
        output of choose_point(), namely [[(x_1,y_1),...]_(m,n), [(x_1,y_1),...]_(m+1,n+1), ...]

    
    ---return
    1. px: 
        [x_m, ..., x_(m+1), ...]
    2. py: 
        [y_n, ..., y_(n+1), ...]
    '''
    
    
    px = []
    py = []
    for i in range(5): #{m,n} ,..., {m+4, n+4} 
        if points[i] == []:
            print ('the (%d, %d) block have no point.' % (i+m, i+n))
            continue
        for j in points[i]:
            px.append(j[0])
            py.append(j[1])
    return px, py

def left_upper(m, n, V, H, points, num_section, delta, percent):
    '''select points located in upper envelope of each rectangles. 
    See appendix of SI to gain theortical detail of this function.
    
    ---parameters
    1. m, n: int
        the same as m, n in choose_point()
    2. points: array
        output of choose_point(), namely [[(x_1,y_1),...]_(m,n), [(x_1,y_1),...]_(m+1,n+1), ...]
    3. num_section: int, default = 3
        didive each rectangle in num_section parts to examine the local cocavity and convexity
        NOTICE: increase this number will slow down the speed of de-noising algorithm
    4. delta: float, >= 0
        see inner functions > f_cave for explaination
    5. percent: float
        see inner functions > local_con for explaination
        
    ---return
    lup = [[(lux_m,luy_n),...], [(lux_(m+1), luy_(n+1)),...], ...] after left_upper
    
    lup denotes "left upper point"
    '''
    #-------inner functions
    def y_diag(i, x):
        '''
        Equation of the diagonal line of {m+i, n+i} rectangle
        '''
        return H[i+n] + (x - V[i+m]) * (H[i+n-1] - H[i+n]) / (V[i+m-1] - V[i+m])
    
    def f_cave(i, x, delta):
        '''
        Parabola equation passes right upper, left bottom, and "middle-high" point of {m+i, n+i}.
        The higher the delta is, the higher the middle point is. In other words, the parabola is more curved.
        '''
        a = 2 * delta * (H[i+n] - H[i+n-1]) / ((V[i+m-1] - V[i+m])**2)
        b = (H[i+n-1] - H[i+n]) / (V[i+m-1] - V[i+m])
        c = H[i+n-1]
        return a * (x - V[i+m-1]) * (x - V[i+m]) + b * (x - V[i+m-1]) + c
    
    def local_con(i, points_set, section_length, percent):
        '''
        This function is used to examine the concavity and convexity (see Eq. (14) and (15) in SI) of local envelope.
        
        1. Each rectangle is divided into num_section sections, so that 
        section_length = x length of {m+i, n+i} rectangle/ num_section
        
        2. percent is used to evaluate how many points need to be selected to represent the local envelope
        
        EX: if percent = 5%, and section_length = 100.
            We need at least 100*5% = 5 points to represent the local envelope in a section.
            
            For each section, we use Eq. (14) to find the modified points {r_0}.
            (a) if we find more than 5 points in this section, it is ok.
            (b) if not, Eq. (15) will be adopted to improve this situation.
            then move on to next section.
        '''
        
        
        points_defog = []
        #when envelope is concave, select them by the following inequality (see Eq. (14) in SI)
        for p in points_set:
            #y_diag(i, p[0]) is the y position of diagonal line in (m, n) rectangle when x = p[0]
            if p[1] >= y_diag(i, p[0]):
                points_defog.append(p)
                
        #points_defog is not enough to represent the local envelope
        if len(points_defog) <= percent * int(section_length): 
            points_defog = []

            #if the envelope is convex, we need to modify it  (see Eq. (15) in SI)
            for p in points_set:
                p_ = p[1] + f_cave(i, p[0], delta) - y_diag(i, p[0])
                if p_ >= y_diag(i, p[0]):
                    points_defog.append(p)

        return points_defog
            
            
    
    #-------------------------------------------------
    lup = [[] for i in range(len(points))]
    for i in range(len(points)): #points in {m+i, n+i} rectangle
        if points[i] == []: #there is no point 
            continue
        
        ##First operation: for each x in {m+i, n+i}, select the highest y and record it
        new_points = [] #also be used to examine the local concavity and convexity
        
        #if severval points share the same x, adopt the one has the highest y value
        x_temp = points[i][0][0]
        y_temp = points[i][0][1]
        
        for j in range(len(points[i])):            
            #record the points share the same x
            if points[i][j][0] == x_temp:
                if points[i][j][1] > y_temp:
                    y_temp = points[i][j][1]
                
            #when x changes, find out y_max and record (x_previous, y_max)
            elif points[i][j][0] < x_temp: #remember that points[i] is reversely sorted according to x value
                new_points.append((x_temp, y_temp))
                x_temp = points[i][j][0] #record x_new
                y_temp = points[i][j][1] #record y_new
        
        #when x doesn't change for the last points, the above program don't record it. So we need to record manually.
        new_points.append((x_temp, y_temp))
        
        
        ##Second operation: examine the local concavity and convexity
        #divide {m+i, n+i} into num_section parts (default: num_section = 10)
        if type(num_section) == int:
            section_length = (V[m+i-1] - V[m+i]) / num_section
        
        
        if num_section == 1:
            local_p = local_con(i, new_points, section_length, percent)
            for k in local_p:
                lup[i].append(k)
        
        elif num_section > 1:
            j = 1
            p_check = []
            for p in new_points:
                if p[0] > V[m+i-1] - j * section_length:
                    p_check.append(p)
                else:
                    local_p = local_con(i, p_check, section_length, percent)
                    for k in local_p:
                        lup[i].append(k)
                    j += 1

            local_p = local_con(i, p_check, section_length, percent)
            for k in local_p:
                lup[i].append(k)
                    
        
    return lup

def Tv(r, p):
    '''
    ---parameters
    r: ndarray
    data points after denosing, where r = [rx1, rx2, ..., rxn, ry1, ry2, ..., ryn]
    p: ndarray
    data with noise, where p = [x1, x2, ..., xn, y1, y2, ..., yn]
    
    ---output
    a function used to denoise
    '''
    def penalty(D, D_0 = 50):
        #This penalty function is used to lower the influence of outliers
        #see book: https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf
        if abs(D) <= D_0:
            return D ** 2
        elif abs(D) > D_0:
            return D * (2 * abs(D) - D_0)
    t = 0
    for i in range(len(p)):
        t = t + penalty(r[i] - p[i], 50)
    
    #see taxicab distance
    Lambda = 1 #regularziation parameters
    rl = int(len(r)/2)
    return sum(np.square(r[1 : rl] - r[:rl - 1])) + sum(np.square(r[rl + 1:] - r[rl:-1])) + Lambda * t

def DENOISE(m, n, V, H, points, toler = 50, num_section = 2, delta = 0.15, percent = 0.05):
    '''chose left_upper part in each rectangle and denoise them via Tv.
    
    ---input
    1. m, n: int
        the same as m, n in choose_point()
    2. points: array
        output of choose_point(), namely [[(x_1,y_1),...]_(m,n), [(x_1,y_1),...]_(m+1,n+1), ...]
    3. toler: number, default = 50
        control the tolerance of minimize(Tv)
    4. num_section: int, default = 3
        see left_upper() for details
        didive each rectangle in num_section parts to examine the local cocavity and convexity
        NOTICE: increase this number will slow down the speed of de-noising algorithm
    5. delta: float, >= 0
        affect the tolerance of left_upper()
        see left_upper() > inner functions > f_cave for explaination
    6. percent: float
        affect the tolerance of left_upper()
        see left_upper() > inner functions > local_con for explaination
    
    ---output
    1. luptx: 1D array
        array of x coordinate for (m,n)~(m+5,n+5) after denoising 
    2. lupty: 1D array
        array of y coordinate for (m,n)~(m+5,n+5) after denoising
    '''
    
    lup = left_upper(m, n, V, H, points, num_section, delta, percent)
    lupx, lupy = sep_point(m, n, lup)
    lupxy = np.array(lupx + lupy)  #lupxy = [x1, x2, ..., xn, y1, y2, ..., yn]
    RR = minimize(Tv, lupxy, args = lupxy, method='CG', tol = toler)
    luptx = RR.x[:int(len(RR.x)/2)]
    lupty = RR.x[int(len(RR.x)/2):]
    return luptx, lupty

def coarse_grain(px, py, Range, num_part = 50):
    '''use coarse-grain on (px,py), ruturn p_avg
    
    ---input
    px, py: 1-D list
    Range: the x-range of data points
    num_part: number of window, i.e. divide pi into num_part parts. 
              the window = (max(Range)-min(Range))/num_part
    
    ---return
    p_avg: (px,py) after coarse-grain
    '''
    window = (max(Range) - min(Range))/num_part
    px_avg = [[] for i in range(num_part)]
    py_avg = [[] for i in range(num_part)]
    p = [(px[i], py[i]) for i in range(len(px))]
    
    for j in p:
        #check every points
        for i in range(num_part):
            if (j[0] >= min(Range) + i*window) & (j[0] <= min(Range) + (i+1)*window):
                px_avg[i].append(j[0])
                py_avg[i].append(j[1])
                break    
    x_avg, y_avg = [], []
    for i in range(num_part):
        if (px_avg[i] == []) or (py_avg[i] == []):
            x_avg.append(float('nan'))
            y_avg.append(float('nan'))
        else:
            x_avg.append(np.mean(px_avg[i]))
            y_avg.append(np.mean(py_avg[i]))        
    return x_avg, y_avg


def plot_g(L, V, H, big, name, longest, toler = 50, num_part = 50, num_section = 2, delta = 0.15, percent = 0.05):
    '''
    ---input
    1. L: integer, 
        this function will select points on scaling line from g_1 to g_L
    2. toler: float, 
        tolerance. Increase this value will speed up the minimization process but decline in performance.
    3. num_part: int
        number of windows (num_part parts) that are used to doing coarse-grain.
    4. num_section: int, default = 2
        see left_upper() for details
        didive each rectangle in num_section parts to examine the local cocavity and convexity
        NOTICE: increase this number will slow down the speed of de-noising algorithm
    5. delta: float, >= 0
        affect the tolerance of left_upper()
        see left_upper() > inner functions > f_cave for explaination
    6. percent: float
        affect the tolerance of left_upper()
        see left_upper() > inner functions > local_con for explaination
        
    
    ---return
    g: set, {g_1, g_2,...,g_L}, where g_k = (x_avg, y_avg) denote the points after coarse-grain
    glu: set, {glu_1, glu_2,...,glu_L}, where glu_k = (lupx, lupy)_k denote the points on scaling line g_k
    
    ps. lu means left upper
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
    for n in range(1, L+1):
        (m, n) = (1, n)
        points = choose_point(m, n, V, H, big, longest)
        px, py = sep_point(m, n, points)
        plt.plot(px, py,'o', markersize = '4')
        if n == 1:
            luptx, lupty = px, py
        else:
            luptx, lupty = DENOISE(m, n, V, H, points, toler, num_section, delta, percent)
        glu['g' + str(n)] = (luptx, lupty)
        plt.plot(luptx, lupty,'.' ,markersize = '4', color = '#e9bf53')
        #coarse-grain
        Range = [0.25*V[0], V[0]]
        x_avg, y_avg = coarse_grain(luptx, lupty, Range, num_part)
        g['g' + str(n)] = (x_avg, y_avg)
    plt.xlim([0, V[0]*1.03])
    plt.ylim([0, H[0]*1.03])
    plt.xlabel('word', size = 15)
    plt.ylabel('syllagram', size = 15)  
    plt.title(name, size = 20)
    plt.show()
    return g, glu

def rg(name, g, FORMAT, Path = ''):
    '''plot r_g of your data
    
    ---parameters
    1. name: str
       name of your r_g plot
    2. g: set, contain points on g1 ~ gL after coarse-grain
       output of plot_g(L, V, H) 
       #PS: use g (coarse-grain data), not glu (original data)
    3. FORMAT: string
       The format of your RRD plot. Most backends support png, pdf, ps, eps and svg. 
       else: just show plot instead of saving.
    
    4. Path: file path for saving picture
       Default: save at current document
       
    ---output
    figure including information about R and error
    
    ---return
    3-D tuple Rg, where
    Rg[0] = R: avg of g_(k+1)/g_k for all k
    Rg[1] = error: standard error for g_(k+1)/g_k
    Rg[2] = R_dist: record g_(k+1)/g_k for every x withour NAN
    
    Rg[0] and Rg[1] are used to calculate SC value, 
    while Rg[2] can conut the distribution of r_g (for future researches, such as error distribution)
    '''
    num_part = max([len(g[i][0]) for i in g])
    len_g = len(g)
    y = {}
    x = {}
    STD = {} #STD of g_n/g_(n+1)
    weight = {} #number of data of g_n/g_(n+1)
    r = {} #average ratio of g_n/g_(n+1)
    R_dist = {} #record g_(k+1)/g_k for every x withour NAN
    for k in range(1, len_g):
        gk1 = 'g' + str(k+1)
        gk = 'g' + str(k)
        y[gk1 + '/' + gk] = [g[gk1][1][j]/g[gk][1][j] for j in range(num_part)]
        x[gk1 + '/' + gk] = [0.5*g[gk1][0][j] + 0.5*g[gk][0][j] for j in range(num_part)]
        
        y_k = [i for i in y[gk1 + '/' + gk] if i==i] # this is y without NAN
        R_dist[gk1 + '/' + gk] = y_k
        STD[gk1 + '/' + gk] = round(np.std(y_k), 3)
        weight[gk1 + '/' + gk] = len(y_k)
        r[gk1 + '/' + gk] = round(np.mean(y_k), 3)
    
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
    marker_list = ['o', 'X', 'D', '^', '<', '>', '1', '2', '3', '4']
    marker_index = 0
    
    for i in x:
        px = x[i]
        py = y[i]
        std = STD[i]
        if i != 'g2/g1':
            C[i] = weight[i]/len(py)
            if C[i] < 0.8:
                print('C < 0.8: %s, %f' % (i, C[i]))
        ax.errorbar(px, py, yerr = std) #plot errorbar
        plt.plot(px, py, marker = marker_list[marker_index], markersize = '4', label = i)
        marker_index += 1
        plt.legend(loc = 'lower left', prop = {'size': 15})
    
    C_value = np.mean([C[i] for i in C])
    S_value = 1 - ERROR/R
    
    xmin, xmax = plt.xlim([0,None])
    ymin, ymax = plt.ylim([0,None])
    
    kwargs = {'fontsize' : 35, 'verticalalignment' : 'bottom', 'horizontalalignment' : 'right'}
    
    plt.text(xmax*0.98, ymax*0.5, '$r_g=%.3f\pm %.3f$' % (R,ERROR), **kwargs)
    plt.text(xmax*0.9, ymax*0.35, '$S=%.3f$' % (S_value), **kwargs)
    plt.text(xmax*0.9, ymax*0.2, '$C=%.3f$' % (C_value), **kwargs)
    plt.text(xmax*0.9, ymax*0.05, '$SC=%.3f$' % (S_value*C_value), **kwargs)
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
    return (R, ERROR, R_dist)

def scaling_fit(data, Rg_0, V, H, Zipf, name, FORMAT = 'pdf', Path = ''):
    '''find out best fitting curve for scaling lines
    use fun to be fitting model, select g1~gN to be basis of scaling function, after that find out the best basis and parameters
    by check deviation of different basis.
    ### Notice: we don't use g1 as basis, and exclude g1 when calculate deviation
    For instance, use g2 to be basis
    0. fitting g2 with fun
    1. g3 = Rg_0*g2, g4 = Rg_0^2*g2......
    2. tot_Dev['g2'] = sum((y_i - g_i)**2), where y_i is real data and g_i is fitting data
    3. check all possible and resonable basis, findout the smallest tot_Dev['gk'] 
    4. calculate fitting score, base on an empirical truth that good fitting use less data
    
    ---parameters
    1. data: set, the output that comes form plot_g(). It can be g or glu
        It is the set of points on scaling lines 
        #PS: I suggest use glu (original data), not g (coarse-grain data)
    2. Rg_0: mean ratio of scaling curve
        It is 0th element of Rg = rg(), i.e., Rg_0 = Rg[0]
        Instead of Rg_0, you can try RH_0 = RH[0] (but fitting performance is bad)
        where RH = which_plot(), RH[0] = mean, RH[1] = std, and RH[2] = shift
    3. Zipf: tuple (c, s), where C is the leading coffecient and s is the exponent of frequency-rank distribution of word
    
    ---output
    a picture with best fitting curve
    
    ---return
    fitting parameters for scaling function
    
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
    def fun_theory(x, q, C):
        '''theory of scaling curve                
        '''
        return q*Rg_0**(C * (x**-Zipf[1]))
    
    def fun(x, q, s):
        '''power law fit
        '''
        return q*x**s
    number = len(data) #number of scaling lines need fitting
    q0 = (100, 0.5) #initial guess
    fit_para = {}
    tot_Dev = {}
    for gk in data:
        if gk != 'g1':           
            theo = {}
            Dev = {}
            #popt is the optimal values for the parameters (q, s, t)
            popt, pcov = curve_fit(fun_theory, data[gk][0], data[gk][1], q0, bounds =(0, [np.inf, np.inf]))
            fit_para[gk] = (popt, pcov)   
            k = int(gk.split('g')[1]) #ex: gk = 'g2' then k = 2
            for i in range(2, number + 1):
                Gi = 'g' + str(i)
                theo[Gi] = fun_theory(data[Gi][0], *popt) * Rg_0**(i - k) #ex: k=2, i=4 then theo['g3'] = fun * Rg^2
                diff = theo[Gi] - data[Gi][1]
                Dev[Gi] = np.sum(np.square(diff))
            tot_Dev[gk] = np.sum([Dev[j] for j in Dev])
        
    best = min(tot_Dev, key = tot_Dev.get) #Get the key corresponding to the minimum value within a dictionary
    k_best = int(best.split('g')[1]) #ex: best = 'g2' then k_best = 2
    
    for i in range(1, number + 1):
        Gi = 'g' + str(i)
        theo[Gi] = fun_theory(data[Gi][0], *fit_para[best][0]) * Rg_0**(i - k_best) #ex: k_best=2, i=1 then theo['g1] = Rg^(-1)* fun
        plt.plot(data[Gi][0], data[Gi][1], '.', markersize = '4', color ='#e9bf53')
        plt.plot(data[Gi][0], theo[Gi], 'o', markersize = '4')
    xm, xM = plt.xlim([0,V[0]*1.03])
    ym, yM = plt.ylim([0,H[0]*1.03])
    
    #-------------------------------------
    '''
    A = format(popt[0], '#.4g')  # give 4 significant digits
    if A[-1] == '.':
        A = A[:-1]
    B = format(popt[1], '#.4g')  # give 4 significant digits
    if B[-1] == '.':
        B = B[:-1]
    #plt.text(0.05*xM, 0.05*yM, r'$f_%d=%s \times %.3f^{%s x^{-%.2f}}$' % (n, A, Rf, B, popt[2]), fontsize=24, color ='black')
    '''
    #-------------------------------------
    
    
    plt.xlabel('word', size = 15)
    plt.ylabel('syllagram', size = 15)  
    plt.title(name, size = 20)
    
    #the following part is used to calculate fitting score, base on an empirical truth that good fitting use less data
    dx_min = {i : min(data[i][0]) for i in data} #find minima x in data
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

def fit_with_cut(data, Rg_0, V, H, Zipf, name, FORMAT, Path = ''):
    '''fit data bigger than 0.25*V[0] to rise accuracy of fitting
    if 0.25*V[0] is not small enough, lowering the low bound of data automatically
    
    ---parameters
    1. data: set, the output that comes form plot_g(). It can be g or glu
        It is the set of points on scaling lines 
        #PS: I suggest use glu (original data), not g (coarse-grain data)
    2. Rg_0: mean ratio of scaling curve
        It is 0th element of Rg = rg(), i.e., Rg_0 = Rg[0]
        Instead of Rg_0, you can try RH_0 = RH[0] (but fitting performance is bad)
        where RH = which_plot(), RH[0] = mean, RH[1] = std, and RH[2] = shift
    3. Zipf: tuple (c, s), where C is the leading coffecient and s is the exponent of frequency-rank distribution of word
    
    ---output
    an fitted plot that has fitting score
    
    ---return
    fit_para: fitting parameters for scaling function
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
            fit_para = scaling_fit(D, Rg_0, V, H, Zipf, name, FORMAT, Path)
            print('fitting range = [%d, %d]' % (dr*V[0], V[0]))
            check = 1
            return fit_para
            break
        except RuntimeError:
            pass
    if check == 0 :
        print('Can not find best parameters in data range.')