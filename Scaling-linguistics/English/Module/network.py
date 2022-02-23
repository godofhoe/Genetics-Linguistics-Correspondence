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
    '''construct the graph of word and syllagram
       G_word: if two words appear in the same syl, there is a edge
       G_syl: if two syls aooear in the same word, they is a edge
       
       paras:
       ------------------------
       coordinate: output from draw_RRD_plot which is defined in Module.count
       
       output:
       ------------------------
       graph_word: list, contains G_word, cluster_word, and word_degree_sequence
       1. G_word: nx.Graph(), constructed by nodes (words) and edges 
       2. cluster_word: nx.clustering(), clustering coefficient of nodes in G_word
       3. word_degree_sequence: G_word.degree(), sequence sorted by degree of nodes in G_word
       
       graph_syl: list, contains G_syl, cluster_syl, and syl_degree_sequence
       1. G_syl: nx.Graph(), constructed by nodes (syllagrams) and edges
       2. cluster_syl: nx.clustering(), clustering coefficient of nodes in G_syl
       3. syl_degree_sequence:  G_syl.degree(), sequence sorted by degree of nodes in G_syl       
    '''    
    
    x, y = [], []
    for i in range(len(coordinate)):
        x.append(coordinate[i][0])
        y.append(coordinate[i][1])

    temp = sorted(coordinate, key = lambda x: x[1])
    edge_word = np.zeros((max(x), max(x)))
    lonely_word = [] #share no syl to other word
    edge_word_list = []

    for j in range(max(y)):
        col = []
        for i in temp:
            if i[1] == j+1:   #word share the same syl
                col.append(i[0])
            elif i[1] > j+1:
                break
        for k in range(len(col)):
            if len(col) == 1:
                lonely_word.append((col[k], col[k]))
            else:
                for m in range(k+1, len(col)):
                    edge_word[col[k]-1][col[m]-1] += 1
                    edge_word[col[m]-1][col[k]-1] += 1
                    edge_word_list.append((col[k], col[m]))
                    
    temp = sorted(coordinate, key = lambda x: x[0])
    edge_syl = np.zeros((max(y), max(y)))
    lonely_syl = []  #share no word to other syl
    edge_syl_list = []
    
    for j in range(max(x)):
        com = [] 
        for i in temp:
            if i[0] == j+1:   #syl share the same word
                com.append(i[1])
            elif i[0] > j+1:
                break
        for k in range(len(com)):
            if len(com) == 1:
                lonely_syl.append((com[k], com[k]))
            else:
                for m in range(k+1, len(com)):
                    edge_syl[com[k]-1][com[m]-1] += 1
                    edge_syl[com[m]-1][com[k]-1] += 1
                    edge_syl_list.append((com[k], com[m]))
                    
    G_word = nx.Graph()
    G_word.add_edges_from(edge_word_list)
    cluster_word = nx.clustering(G_word)
    word_degree_sequence = sorted([d for n, d in G_word.degree()], reverse=True)  # degree sequence

    graph_word = (G_word, cluster_word, word_degree_sequence)
    
    G_syl = nx.Graph()
    G_syl.add_edges_from(edge_syl_list)
    cluster_syl = nx.clustering(G_syl)
    syl_degree_sequence = sorted([d for n, d in G_syl.degree()], reverse=True)  # degree sequence
    
    graph_syl = (G_syl, cluster_syl, syl_degree_sequence)
    return graph_word, graph_syl

def plot_cluster_word(name, cluster_word, FORMAT = 'pdf', Path = ''):
    '''
    calculate the local clustering coefficient and cumulative probability of word layer
    '''

    fig, ax = plt.subplots()

    cluster_rank_word = list(cluster_word)
    cluster_coef_word = sorted([cluster_word[i] for i in cluster_word], reverse = True)
    plt.title('Cumulative probability of $C_b$', fontsize = 20)
    plt.xlabel('Local clustering coefficient $C_b$', size = 20)
    plt.ylabel('$P(x|x\leq C_b)$', size = 20)
    ax.tick_params(axis='x', labelsize=15) 
    ax.tick_params(axis='y', labelsize=15)
    #https://stackoverflow.com/questions/6774086/why-is-my-xlabel-cut-off-in-my-matplotlib-plot
    plt.gcf().subplots_adjust(left = 0.17, bottom = 0.17)
    plt.hist(cluster_coef_word, bins = 20, cumulative=True, density = 1)
    ym, yM = plt.ylim()
    xm, xM = plt.xlim()
    plt.text(xM/10+xm*9/10, yM*4/5, 'average $C_b$=%.2f' % (sum(cluster_coef_word)/len(cluster_coef_word)), fontsize=30)
    try:
        if Path == '':
            fig.savefig('cluster_word_' + name + '.' + FORMAT, dpi = 400, format = FORMAT)
            plt.show()
        else:
            fig.savefig(Path + 'cluster_word_' + name + '.' + FORMAT, dpi=400, format = FORMAT)
            plt.close()
    except:
        plt.show()
        
def plot_cluster_syl(name, cluster_syl, FORMAT = 'pdf', Path = ''):
    '''
    calculate the local clustering coefficient and cumulative probability of syllagram layer
    '''

    fig, ax = plt.subplots()

    cluster_rank_syl = list(cluster_syl)
    cluster_coef_syl = sorted([cluster_syl[i] for i in cluster_syl], reverse = True)
    plt.title('Cumulative probability of $C_c$', fontsize = 20)
    plt.xlabel('Local clustering coefficient $C_c$', size = 20)
    plt.ylabel('$P(x|x\leq C_c)$', size = 20)
    ax.tick_params(axis='x', labelsize=15) 
    ax.tick_params(axis='y', labelsize=15)
    #https://stackoverflow.com/questions/6774086/why-is-my-xlabel-cut-off-in-my-matplotlib-plot
    plt.gcf().subplots_adjust(left = 0.17, bottom = 0.17)
    plt.hist(cluster_coef_syl, bins = 20, cumulative=True, density = 1)
    ym, yM = plt.ylim()
    xm, xM = plt.xlim()
    plt.text(xM/6+xm*5/6, yM/2, 'average $C_c$=%.2f' % (sum(cluster_coef_syl)/len(cluster_coef_syl)), fontsize=30)
    try:
        if Path == '':
            fig.savefig('cluster_syl_' + name + '.' + FORMAT, dpi = 400, format = FORMAT)
            plt.show()
        else:
            fig.savefig(Path + 'cluster_syl_' + name + '.' + FORMAT, dpi=400, format = FORMAT)
            plt.close()
    except:
        plt.show()

def plot_degree_word(name, word_degree_sequence, FORMAT = 'pdf', Path = ''):
    D = count_frequency(word_degree_sequence)
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
            fig.savefig('degree_word_' + name + '.' + FORMAT, dpi = 400, format = FORMAT)
            plt.show()      
        else:
            fig.savefig(Path + 'degree_word_' + name + '.' + FORMAT, dpi = 400, format = FORMAT)
            plt.close()
    except:
        plt.show()

def plot_degree_syl(name, syl_degree_sequence, FORMAT = 'pdf', Path = ''):
    #use MLE to get the fitting parameter, detial read: Curve_Fitting_MLE
    D = count_frequency(syl_degree_sequence)
    #T = ([degree], [degreeFreq]) #we don't fit those which degree = 0
    T = ([], [])
    for i in D:
        if i != 0:
            T[0].append(i)
            T[1].append(D[i])
    Y = Two_to_One(T)

    xdata = np.linspace(min(T[0]), max(T[0]), num = (max(T[0]) - min(T[0]))*10)

    #y(x) = Cx^(-s)
    #res = minimize(L_Zipf, (3), Y, method = 'CG')
    #s = res['x'][0]
    #t = [int(min(T[0])), int(max(T[0])), s]
    #C = 1 / incomplete_harmonic(t)
    #theo = Zipf_law(xdata, s, C) #Notice theo is normalized, i.e, the probability density


    #y(x) = C(x + a)^(-s)
    res = minimize(L_Zipf_Mandelbrot, (3, 0), Y, method = 'CG')
    s = res['x'][0]
    t = [int(min(T[0])), int(max(T[0])), s]
    a = res['x'][1]
    C = 1 / incomplete_shifted_harmonic(t, a)
    theo = Zipf_Mandelbrot(xdata, s, C, a) #Notice theo is normalized, i.e, the probability density


    N = sum(T[1])
    theo = [N * i for i in theo] #change theo from probability density to real frequency

    fig, ax = plt.subplots()
    plt.plot(T[0], T[1], 'ro', markersize=4)
    plt.plot(xdata, theo, 'g-')

    #the following code deal with significant figures of fitting parameters
    #the tutor of significant figures: https://www.usna.edu/ChemDept/_files/documents/manual/apdxB.pdf
    #-----------------------------------------
    x_dig = len(str(max(T[0]))) 
    y_dig = len(str(max(T[1])))
    s_dig = min(x_dig+1, y_dig +1) #significant figures of exponent s, it comes from log(y)/log(x)
    a_dig = x_dig #significant figures of parameter a

    # the fomat string is #.?g, where ? = significant figures
    # detail of the fomat string: https://bugs.python.org/issue32790
    # https://docs.python.org/3/tutorial/floatingpoint.html
    S = format(s, '#.%dg' % s_dig)  # give a_dig significant digits
    A = format(a, '#.%dg' % a_dig)  # give b_dig significant digits
    if 'e' in S: #make scientific notation more beautiful
        S = S.split('e')[0] + '\\times 10^{' + str(int(S.split('e')[1])) + '}'
    if S[-1] == '.':
        S = S[:-1]
    if 'e' in A: #make scientific notation more beautiful
        A = A.split('e')[0] + '\\times 10^{' + str(int(A.split('e')[1])) + '}'
    if A[-1] == '.':
        A = A[:-1]

    parameters = (r"$d_0=%s$"
                      "\n"
                     r"$\eta=%s$") % (A, S)    

    text_x = 1.5      
    text_y = min(theo)
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
            fig.savefig('degree_syl_' + name + '.' + FORMAT, dpi = 400, format = FORMAT)
            plt.show()      
        else:
            fig.savefig(Path + 'degree_syl_' + name + '.' + FORMAT, dpi = 400, format = FORMAT)
            plt.close()
    except:
        plt.show()        
        
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