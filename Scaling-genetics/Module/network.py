# -*- coding: utf-8 -*-
'''
@author  gmking

This module contains the tools to construct and analyze network
'''

import networkx as nx
import time
import random 
import bisect 
import math 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Module.count import *
from Module.Curve_Fitting_MLE import *
from scipy.optimize import curve_fit

def build_edge(coordinate):
    '''construct the graph of block and component
       G_block: if two blocks appear in the same compo, there is a edge
       G_compo: if two components appear in the same block, they is a edge
       
       ---Input
           coordinate: output from draw_RRD_plot() which is defined in Module.count
       
       ---Return
       1. graph_block: list, contains G_block, cluster_block, and block_degree_sequence
           (1) G_block: nx.Graph(), constructed by nodes (blocks) and edges 
           (2) cluster_block: nx.clustering(), clustering coefficient of nodes in G_block
           (3) block_degree_sequence: G_block.degree(), sequence sorted by degree of nodes in G_block
       
       2. graph_compo: list, contains G_compo, cluster_compo, and compo_degree_sequence
           (1) G_compo: nx.Graph(), constructed by nodes (components) and edges
           (2) cluster_compo: nx.clustering(), clustering coefficient of nodes in G_compo
           (3) compo_degree_sequence:  G_compo.degree(), sequence sorted by degree of nodes in G_compo       
    '''    
    
    x, y = [], []
    for i in range(len(coordinate)):
        x.append(coordinate[i][0])
        y.append(coordinate[i][1])

    temp = sorted(coordinate, key = lambda x: x[1])
    edge_block = np.zeros((max(x), max(x)))
    lonely_block = [] #share no components to other block
    edge_block_list = []

    for j in range(max(y)):
        col = []
        for i in temp:
            if i[1] == j+1:   #block share the same components
                col.append(i[0])
            elif i[1] > j+1:
                break
        for k in range(len(col)):
            if len(col) == 1:
                lonely_block.append((col[k], col[k]))
            else:
                for m in range(k+1, len(col)):
                    edge_block[col[k]-1][col[m]-1] += 1
                    edge_block[col[m]-1][col[k]-1] += 1
                    edge_block_list.append((col[k], col[m]))
                    
    temp = sorted(coordinate, key = lambda x: x[0])
    edge_compo = np.zeros((max(y), max(y)))
    lonely_compo = []  #share no block to other components
    edge_compo_list = []
    
    for j in range(max(x)):
        com = [] 
        for i in temp:
            if i[0] == j+1:   #components share the same block
                com.append(i[1])
            elif i[0] > j+1:
                break
        for k in range(len(com)):
            if len(com) == 1:
                lonely_compo.append((com[k], com[k]))
            else:
                for m in range(k+1, len(com)):
                    edge_compo[com[k]-1][com[m]-1] += 1
                    edge_compo[com[m]-1][com[k]-1] += 1
                    edge_compo_list.append((com[k], com[m]))
                    
    G_block = nx.Graph()
    G_block.add_edges_from(edge_block_list)
    cluster_block = nx.clustering(G_block)
    block_degree_sequence = sorted([d for n, d in G_block.degree()], reverse=True)  # degree sequence

    graph_block = (G_block, cluster_block, block_degree_sequence)
    
    G_compo = nx.Graph()
    G_compo.add_edges_from(edge_compo_list)
    cluster_compo = nx.clustering(G_compo)
    compo_degree_sequence = sorted([d for n, d in G_compo.degree()], reverse=True)  # degree sequence
    
    graph_compo = (G_compo, cluster_compo, compo_degree_sequence)
    return graph_block, graph_compo

def plot_cluster_block(name, cluster_block, FORMAT = 'pdf', Path = ''):
    '''
    calculate the local clustering coefficient and cumulative probability of block layer
    
    ---Input
    1. name: string
        name of the plot
        
    2. cluster_block: graph_block[2]
        graph_block is return of build_edge
        
    3. FORMAT: string
        The format of your plot. Most backends support png, pdf, ps, eps and svg. 
        else: just show plot instead of saving.
    
    4. Path: file path for saving picture
        Default: save at current document
        if Path == np.nan, no figure will be saved (just show it)
        else, the figure will be saved according to Path
        
    ---Output
        save or show a figure of local clustering coefficient and cumulative probability of block layer
    '''

    fig, ax = plt.subplots()

    cluster_rank_block = list(cluster_block)
    cluster_coef_block = sorted([cluster_block[i] for i in cluster_block], reverse = True)
    plt.title('Cumulative probability of $C_b$', fontsize = 20)
    plt.xlabel('Local clustering coefficient $C_b$', size = 20)
    plt.ylabel('$P(x|x\leq C_b)$', size = 20)
    ax.tick_params(axis='x', labelsize=15) 
    ax.tick_params(axis='y', labelsize=15)
    #https://stackoverflow.com/questions/6774086/why-is-my-xlabel-cut-off-in-my-matplotlib-plot
    plt.gcf().subplots_adjust(left = 0.17, bottom = 0.17)
    plt.hist(cluster_coef_block, bins = 20, cumulative=True, density = 1)
    ym, yM = plt.ylim()
    xm, xM = plt.xlim()
    plt.text(xM/10+xm*9/10, yM*4/5, 'average $C_b$=%.2f' % (sum(cluster_coef_block)/len(cluster_coef_block)), fontsize=30)
    try:
        if Path == '':
            fig.savefig('cluster_block_' + name + '.' + FORMAT, dpi = 400, format = FORMAT)
            plt.show()
        else:
            fig.savefig(Path + 'cluster_block_' + name + '.' + FORMAT, dpi=400, format = FORMAT)
            plt.close()
    except:
        plt.show()
        
def plot_cluster_compo(name, cluster_compo, FORMAT = 'pdf', Path = ''):
    '''
    calculate the local clustering coefficient and cumulative probability of component layer
    
    ---Input
    1. name: string
        name of the plot
        
    2. cluster_compo: graph_compo[2]
        graph_compo is return of build_edge
        
    3. FORMAT: string
        The format of your plot. Most backends support png, pdf, ps, eps and svg. 
        else: just show plot instead of saving.
    
    4. Path: file path for saving picture
        Default: save at current document
        if Path == np.nan, no figure will be saved (just show it)
        else, the figure will be saved according to Path
        
    ---Output
        save or show a figure of local clustering coefficient and cumulative probability of block layer
    '''

    fig, ax = plt.subplots()

    cluster_rank_compo = list(cluster_compo)
    cluster_coef_compo = sorted([cluster_compo[i] for i in cluster_compo], reverse = True)
    plt.title('Cumulative probability of $C_c$', fontsize = 20)
    plt.xlabel('Local clustering coefficient $C_c$', size = 20)
    plt.ylabel('$P(x|x\leq C_c)$', size = 20)
    ax.tick_params(axis='x', labelsize=15) 
    ax.tick_params(axis='y', labelsize=15)
    #https://stackoverflow.com/questions/6774086/why-is-my-xlabel-cut-off-in-my-matplotlib-plot
    plt.gcf().subplots_adjust(left = 0.17, bottom = 0.17)
    plt.hist(cluster_coef_compo, bins = 20, cumulative=True, density = 1)
    ym, yM = plt.ylim()
    xm, xM = plt.xlim()
    plt.text(xM/6+xm*5/6, yM/2, 'average $C_c$=%.2f' % (sum(cluster_coef_compo)/len(cluster_coef_compo)), fontsize=30)
    try:
        if Path == '':
            fig.savefig('cluster_compo_' + name + '.' + FORMAT, dpi = 400, format = FORMAT)
            plt.show()
        else:
            fig.savefig(Path + 'cluster_compo_' + name + '.' + FORMAT, dpi=400, format = FORMAT)
            plt.close()
    except:
        plt.show()

def plot_degree_block(name, block_degree_sequence, FORMAT = 'pdf', Path = ''):
    D = count_frequency(block_degree_sequence)
    #T = ([degree], [degreeFreq]) #we don't fit those points with degree = 0
    T = ([], [])
    for i in D:
        if i != 0:
            T[0].append(i)
            T[1].append(D[i])
            
    fig, ax = plt.subplots()
    plt.plot(T[0], T[1], 'ro', markersize=4)
    plt.xlabel('Degree $d_b$', size = 25)
    plt.ylabel('Frequency', size = 25)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.title(name, fontsize = 25)
    
    #https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tick_params.html
    plt.tick_params(which = 'major', length = 10)
    plt.tick_params(which = 'minor', length = 4)
    
    #https://stackoverflow.com/questions/6774086/why-is-my-xlabel-cut-off-in-my-matplotlib-plot
    plt.gcf().subplots_adjust(left = 0.17, bottom = 0.17)
    try:
        if Path == '':
            fig.savefig('degree_block_' + name + '.' + FORMAT, dpi = 400, format = FORMAT)
            plt.show()      
        else:
            fig.savefig(Path + 'degree_block_' + name + '.' + FORMAT, dpi = 400, format = FORMAT)
            plt.close()
    except:
        plt.show()

def plot_degree_compo(name, compo_degree_sequence, FORMAT = 'pdf', Path = ''):
    '''draw degree distribution of component network

    ---Parameters
    1. name: str
       "XXX" (your file name without filename extension)

    2. compo_degree_sequence: list
       one return of build_edge()
       

    3. FORMAT: string
        The format of your plot. Most backends support png, pdf, ps, eps and svg. 
        else: just show plot instead of saving.
    
    4. Path: file path for saving picture
        Default: save at current document
        if Path == np.nan, no figure will be saved (just show it)
        else, the figure will be saved according to Path
    
    ---Output
        save or show a figure of degree distribution
    
    ---Return:
        degree_component: dict, where
        (1) degree_component['abc']: tuple (a_ZM, b_ZM, c_ZM)
                parameters of P(x, b, c) = a_ZM*(x + c_ZM)^-b_ZM
        (2) degree_component['bc_jac']: tuple
                gradient vector used for optimization
        (3) degree_component['neg_L']: float
                negative max liklihood. 
                details see Curve_Fitting_MLE.py > L_Zipf_Mandelbrot()
        
    '''
    #use MLE to get the fitting parameter, detial read: Curve_Fitting_MLE
    D = count_frequency(compo_degree_sequence)
    #T = ([degree], [degreeFreq]) #we don't fit those which degree = 0
    T = ([], [])
    for i in D:
        if i != 0:
            T[0].append(i)
            T[1].append(D[i])
    Y = Two_to_One(T)
    
    xdata = np.linspace(min(T[0]), max(T[0]), num = (max(T[0]) - min(T[0]))*10)
    
    #Estimate exponent. This action can make reduce the error of initial value guess.
    freq_M, freq_m = max(T[1]), min(T[1])
    rank_M, rank_m = max(T[0]), min(T[0])
    b_0 = np.log(freq_M / freq_m) / np.log(rank_M / rank_m)

    #fit Zipf-Mandelbrot: P(x, b_ZM, c_ZM)=a_ZM/(x + c_ZM)^b_ZM
    res_ZM = minimize(L_Zipf_Mandelbrot, (b_0, 0), Y)
    b_ZM = float(res_ZM['x'][0])
    c_ZM = float(res_ZM['x'][1])
    t_ZM = [int(min(T[0])), int(max(T[0])), b_ZM]
    a_ZM = float(1 / incomplete_shifted_harmonic(t_ZM, c_ZM))
    
    degree_component = {}
    degree_component['abc'] = (a_ZM, b_ZM, c_ZM)
    degree_component['bc_jac'] = tuple(res_ZM.get('jac'))
    degree_component['neg_L'] = res_ZM.get('fun')
    
    #change theo from probability density to real frequency
    N = sum(T[1])
    theo_ZM = N * Zipf_Mandelbrot(xdata, a_ZM, b_ZM, c_ZM)

    fig, ax = plt.subplots()
    plt.plot(T[0], T[1], 'ro', markersize=4)
    plt.plot(xdata, theo_ZM, 'g-')

    #the following code deal with significant figures of fitting parameters
    #the tutor of significant figures: https://www.usna.edu/ChemDept/_files/documents/manual/apdxB.pdf
    #-----------------------------------------
    x_dig = len(str(max(T[0]))) 
    y_dig = len(str(max(T[1])))
    b_dig = min(x_dig+1, y_dig +1) #significant figures of exponent b, it comes from log(y)/log(x)
    c_dig = x_dig #significant figures of parameter c (shift of x)

    # the fomat string is #.?g, where ? = significant figures
    # detail of the fomat string: https://bugs.python.org/issue32790
    # https://docs.python.org/3/tutorial/floatingpoint.html
    B = format(b_ZM, '#.%dg' % b_dig)  # give b_dig significant digits
    C = format(c_ZM, '#.%dg' % c_dig)  # give c_dig significant digits
    if 'e' in C: #make scientific notation more beautiful
        C_text = C.split('e')[0] + '\\times 10^{' + str(int(C.split('e')[1])) + '}'
    elif C[-1] == '.':
        C_text = C[:-1]
    else:
        C_text = C
    if 'e' in B: #make scientific notation more beautiful
        B_text = B.split('e')[0] + '\\times 10^{' + str(int(B.split('e')[1])) + '}'
    elif B[-1] == '.':
        B_text = B[:-1]
    else:
        B_text = B
    #-----------------------------------------

    parameters = (r"$d_0=%s$"
                      "\n"
                     r"$\eta=%s$") % (C_text, B_text)    

    text_x = 1.5      
    text_y = min(theo_ZM)
    plt.text(text_x, text_y, parameters, fontsize = 30)

    plt.xlabel('Degree $d_c$', size = 25)
    plt.ylabel('Frequency', size = 25)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.title(name, fontsize = 25)
    
    #https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tick_params.html
    plt.tick_params(which = 'major', length = 10)
    plt.tick_params(which = 'minor', length = 4)
    
    #https://stackoverflow.com/questions/6774086/why-is-my-xlabel-cut-off-in-my-matplotlib-plot
    plt.gcf().subplots_adjust(left = 0.17, bottom = 0.17)
    try:
        if Path == '':
            fig.savefig('degree_compo_' + name + '.' + FORMAT, dpi = 400, format = FORMAT)
            plt.show()      
        else:
            fig.savefig(Path + 'degree_compo_' + name + '.' + FORMAT, dpi = 400, format = FORMAT)
            plt.close()
    except:
        plt.show()
    return degree_component
        
def build_shortest_path(graph):
    pathL = nx.shortest_path_length(graph)  #shorest path length
    #in networkx 2.0, the structure of pathL will be 
    #[(source_A, {target_B: distance, target_C: distance}), (source_B, {...}), ...]
    sum_shorest_path= []
    for source_i, target_set in pathL: #target_set = {target: distance, target: distance, ...}
        sp_i = 0 #sum_shorest_path for node i
        for j in target_set:
            sp_i += target_set[j]
        sum_shorest_path.append(sp_i)    
    return sum_shorest_path


def plot_shortest_path(name, sum_shorest_path, G_name, FORMAT = 'pdf', Path = ''):
    '''
    sum_shorest_path: list
        output of build_shortest_path
    G_name: str
        name of graph, can use latex form like $G_b$ (for block) or $G_c$ (for component)
    '''
    nonzero_sp = []
    for i in sum_shorest_path:
        if i !=0:
            nonzero_sp.append(i)
    n_nonzero = len(nonzero_sp)
    avg_spL = sum(nonzero_sp)/(n_nonzero*(n_nonzero-1))
    n = len(sum_shorest_path)
    sp = [i/n for i in sum_shorest_path]

    fig, ax = plt.subplots()
    plt.hist(sp, bins = 20)
    plt.xlabel('Path length in %s' % G_name, size = 20)
    plt.ylabel('Frequency', size = 20)
    plt.yscale('log')
    ax.tick_params(axis='x', labelsize=15) 
    ax.tick_params(axis='y', labelsize=15)
    #https://stackoverflow.com/questions/6774086/why-is-my-xlabel-cut-off-in-my-matplotlib-plot
    plt.gcf().subplots_adjust(left = 0.17, bottom = 0.17)
    plt.title('Distribution of path length', fontsize = 20)
    ym, yM = plt.ylim()
    xm, xM = plt.xlim()
    plt.text(xM/6+xm*5/6, yM/2, 'Average without zero=%.2f' % avg_spL, fontsize=20)
    try:
        if Path == '':
            fig.savefig('path_' + G_name + '_' + name + '.' + FORMAT, dpi = 400, format = FORMAT)
            plt.show()
        else:
            fig.savefig(Path + 'path_' + G_name + '_' + name + '.' + FORMAT, dpi = 400, format = FORMAT)
            plt.close()
    except:
        plt.show()
    print('exclude zero path, the avg_path is', avg_spL)