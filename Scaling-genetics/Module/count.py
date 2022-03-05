# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 17:02:47 2016

@author: shan, gmking

This module is used to construct a dataframe with all information we need.
The core function of this module is info()

The main quests are 
1. construct Book
2. build FRD, RRD, {V_m} and {H_n}
3. count distribution of N domains
4. save data
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import ticker
import textwrap
import sys
from .Curve_Fitting_MLE import *
from .zipfgen import ZipfGenerator #https://medium.com/pyladies-taiwan/python-%E7%9A%84-import-%E9%99%B7%E9%98%B1-3538e74f57e3
import random


def read_file(filename, encode = 'UTF-8'):
    """
    Read the text file with the given filename;
    return a list of the proteins of text in the file; ignore punctuations.
    also returns the longest protein length in the file.
    
    ---Input
        a txt file with 'encode' encoding
    
    ---Parameters
    1. file_name : string
          XXX.txt. We suggest you using the form that set 
          name = 'XXX' 
          and 
          filename = name + '.txt'.

    2. encode : encoding of your txt
    
    ---Return
    1. protein_list: array
        a list of proteins in the txt file
        
    2. longest: int
        max number of domains in a protein
        
    """
    punctuation_set = set(u'''_—＄％＃＆:#$&!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒
    ﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠
    々‖•·ˇˉ－―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻
    ︽︿﹁﹃﹙﹛﹝（｛“‘-—_…''')
    longest = 0
    protein_list = []
    with open(filename, "r", encoding = encode) as file:
        for line in file:
            l = line.split()
            for protein in l:
                new_protein = ''
                for c in protein:
                    if c not in punctuation_set:
                        new_protein = new_protein + c
                if not len(new_protein) == 0: 
                    protein_list.append(protein)
                    if len(protein.split('-')) > longest:
                        longest = len(protein.split('-')) #max number of domains in a protein
                    
    if '\ufeff' in protein_list:
        protein_list.remove('\ufeff')
        
    print("read file successfully!")
    return protein_list, longest

def read_Ngram_file(filename, N, encode = 'UTF-8'):
    """
    Read the text file with the given filename;    
    return a list of the proteins of text in the file; ignore punctuations.
    also returns the longest protein length in the file.
    
    ---Parameters
    1. file_name : string
      XXX.txt. We suggest you using the form that set 
      name = 'XXX' 
      and 
      filename = name + '.txt'.
        
    2. N: int
      "N"-gram. 
      For example : a string, ABCDEFG (In Chinese, you don't know what's the 'proteins' of a string)
      in 2-gram = [AB, CD, EF, G];
      in 3-gram = [ABC, DEF, G];
      in 4-gram = [ABCD, EFG]
      
      two proteins are in a txt 'ABCDE EFGHI' (This case happended in English corpus)
      in 2-gram = [AB, CD, E, EF, GH, I];
      in 3-gram = [ABC, DE, EFG, HI];
      in 4-gram = [ABCD, E, EFGH, I]
    
    3. encode : encoding of your txt
    
    ---Return
    1. protein_list: array
        a list of proteins in the txt file
        
    2. N: int
        N-gram
    """
    punctuation_set = set(u'''_—＄％＃＆:#$&!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒
    ﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠
    々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻
    ︽︿﹁﹃﹙﹛﹝（｛“‘-—_…''')

    protein_list = []
    with open(filename, "r", encoding = encode) as file:
        for line in file:
            l = line.split()
            for protein in l:
                new_protein = ''
                for c in protein:
                    if c not in punctuation_set:
                        new_protein = new_protein + c
                    if c in punctuation_set and len(new_protein) != 0:
                        protein_list.append(new_protein)
                        new_protein = ''
                if not len(new_protein) == 0: 
                    protein_list.append(new_protein)
                    new_protein = ''
                    
    if '\ufeff' in protein_list:
        protein_list.remove('\ufeff')
    
    protein_list = []
    for protein in protein_list:
        domains = textwrap.wrap(protein, N)
        New_protein = ''
        for s in domains:
            New_protein = New_protein + '-' + s
            protein_list.append(New_protein)
    
    print("read file successfully!")
    return protein_list, N


def count_frequency(unit_list):
    """count the frequency of occurrence for unit in unit_list
    
    ---Input 
        unit_list: list
            a list containing proteins or domains
        
    ---Return: 
        D: set
            a dictionary mapping proteins to frequency.
    """
    D = {}
    for new_unit in unit_list:
        if new_unit in D:
            D[new_unit] = D[new_unit] + 1
        else:
            D[new_unit] = 1
    return D   


def decide_seq_order(unit_list):
    """generate the seqence of order of units in unit_list
    
    ---Input
        unit_list: list
            unit = protein or domain
            a list containing units

    ---Return 
    1. D: set
        a dictionary mapping each unit to its sequential number, 
        which is decided by the order of first appearence in the unit_list.
        
    2. another_list: list
        a list containg non-repetitive proteins obeys the order of first appearence in the unit_list
    """
    D = {}
    another_list = []
    num = 0
    
    for unit in unit_list:
        if unit not in another_list:
            another_list.append(unit)
            D[another_list[num]] = num + 1
            num += 1
    
    return D, another_list


def transfrom_proteinlist_into_domlist(protein_list):
    """Divide each protein in the protein_list into domains, order preserved
    
    ---Input
        a list contains proteins
    
    ---Return
        a list contains domains
    """
    dom_list = []
    for protein in protein_list:
        dom_list.extend(protein.split('-'))
        
    return dom_list

'''
def transfrom_proteinlist_into_Ngram_domlist(protein_list, N):
    """Divide each proteins in the protein_list into N-gram domains, order reserved.
    
    -------Input: 
    1. protein_list:
      a list containing proteins
    
    2. N: int
      "N"-gram. 
      For example : a string, ABCDEFG (In Chinese, you don't know what's the 'proteins' of a string)
      in 2-gram = [AB, CD, EF, G];
      in 3-gram = [ABC, DEF, G];
      in 4-gram = [ABCD, EFG]
      
      two proteins are in a txt 'ABCDE EFGHI' (This case happended in English corpus)
      in 2-gram = [AB, CD, E, EF, GH, I];
      in 3-gram = [ABC, DE, EFG, HI];
      in 4-gram = [ABCD, E, EFGH, I]
    
    -------Return:
        a list containg dom 
    """
    dom_list = []
    for protein in protein_list:
        domains = textwrap.wrap(protein, N)
        dom_list.extend(domains)
        
    return dom_list
'''

def produce_data_frame(unit_list, unit_freq, unit_seq, varibleTitle):
    '''produce pandas dataframe
    
    ---Input
    unit denotes protein or domain 
    
    1. unit_list: list
        return of read_file() or transfrom_proteinlist_into_domlist()
    
    2. unit_freq: set
        return of count_frequency()
    
    3. unit_seq: set
        return of decide_seq_order()
    
    ---Parameters
        varibleTitle: string
            the title name of column in dataframe
    
    ---Return
        dataframe: pandas.DataFrame
            a dataframe contains unit_list, unit_freq, and unit_seq
    '''
    
    
    unit_list = list(set(unit_list))
    data = {}
    unit_seq_list = []
    unit_freq_list = []
    
    for unit in unit_list:
        unit_freq_list.append(unit_freq[unit])
        unit_seq_list.append(unit_seq[unit])
    
    first = varibleTitle 
    second = varibleTitle + "SeqOrder"
    third = varibleTitle + "Freq"
    forth = varibleTitle + "Rank"
    
    data[first] = unit_list
    data[second] = unit_seq_list
    data[third] = unit_freq_list  
    
    dataFrame = pd.DataFrame(data)
    dataFrame = dataFrame.sort_values([third, second],ascending = [False,True])
    rank = np.array(list(range(1,len(dataFrame)+1))) 
    dataFrame[forth] = rank
    column_list = [first, third, forth, second]
    dataFrame = dataFrame[column_list]
    dataFrame = dataFrame.reset_index(drop=True)
    return dataFrame


def produce_proteinRank_domRank_frame(pd_protein, pd_dom, longest):
    '''merge pd_protein and pd_dom into a large dataframe
    
    ---Input
    1. pd_protein: pandas.DataFrame
        return of produce_data_frame(unit = protein)
    
    2. pd_dom: pandas.DataFrame
        return of produce_data_frame(unit = domain)
    
    3. longest: int
        return of read_file
        
    ---Return
        pd_protein: pandas.DataFrame
            a large dataframe contains information of proteins and its domains
            Ex: for 'ap-ple'
                1th_dom = ap
                2th_dom = ple
    
    '''
    
    
    D = {}
    
    dom_array = pd_dom["dom"]
    dom_rank = {}
    
    for i in range(len(pd_dom)):
        dom_rank[dom_array[i]] = i + 1 
    
    for i in range(longest):
        D[i] = []
    
    protein_array = pd_protein["protein"]
    N_domain = [] #count how many domain in a protein
    
    for protein in protein_array:
        t = protein.split('-') #t is number of domain
        N_domain.append(len(t))
        
        for i in range(len(t)):
            D[i].append(int(dom_rank[t[i]]))
        
        if len(t) < longest:
            for j in range(len(t),longest):
                D[j].append(np.nan)
    
    pd_protein["N_dom"] = np.array(N_domain)
    
    for k in range(longest):
        feature = str(k) + "th" + "_dom_rank"
        pd_protein[feature] = np.array(D[k])
    
    return pd_protein  


def info(file_name, encode = "UTF-8"):
    '''the core function that give you statistical data, including
    1. a dataframe contains proteins and their all domains (big)
    2. the frequency information of domains (dom) and proteins (protein)
    3. the longest length of single protein (longest)
        
    ---Parameters
    1. file_name : string, it must be like XXX.txt
      We suggest you using the form that set 
      name = 'XXX' and filename = name + '.txt'.
    
    2. encode : encoding of your txt
    
    
    ---Return
    1. data_frame: pandas.DataFrame
      a big data frame contains the information of proteins and its domains
      
    2. pd_dom: pandas.DataFrame
      a data frame contains the frequency information of domains
      
    3. another_protein: pandas.DataFrame
      a data frame contains the frequency information of proteins
      
    4. longest_L: int
      the biggest length of single protein.
    
    '''
    L, longest_L = read_file(file_name, encode)
    protein_freq = count_frequency(L)
    print("Successfully count protein freqency!" + "(%s)" % file_name)
    
    protein_seq, protein_list = decide_seq_order(L)
    c_list = transfrom_proteinlist_into_domlist(L)
    dom_seq, dom_list = decide_seq_order(c_list)
    dom_freq = count_frequency(c_list)
    print("Successfully count dom freqency!")
    
    pd_protein= produce_data_frame(protein_list, protein_freq, protein_seq,"protein")
    another_protein = pd_protein.copy()
    pd_dom= produce_data_frame(dom_list, dom_freq, dom_seq,"dom")
    data_frame = produce_proteinRank_domRank_frame(pd_protein, pd_dom, longest_L)
    print("Successfully build data frames!")
    
    return data_frame, pd_dom, another_protein, longest_L

def N_gram_info(file_name, N, encode = "UTF-8"):
    '''This is only used to analysis N-gram proteins.
        
    ---Parameters
    1. file_name : string
      XXX.txt. We suggest you using the form that set 
      name = 'XXX' and filename = name + '.txt'.
        
    2. N: int
      "N"-gram. 
      For example : a string, ABCDEFG (In Chinese, you don't know what's the 'proteins' of a string)
      in 2-gram = [AB, CD, EF, G];
      in 3-gram = [ABC, DEF, G];
      in 4-gram = [ABCD, EFG]
      
      two proteins are in a txt 'ABCDE EFGHI' (This case happended in English corpus)
      in 2-gram = [AB, CD, E, EF, GH, I];
      in 3-gram = [ABC, DE, EFG, HI];
      in 4-gram = [ABCD, E, EFGH, I]
    
    3. encode : encoding of your txt
    
    
    ---Return
    1. data_frame: pandas.DataFrame
      a big data frame contains the information of proteins and its domains
      
    2. pd_dom: pandas.DataFrame
      a data frame contains the information of domains
    
    3. another_protein: pandas.DataFrame
      a data frame contain the information of proteins
      
    4. longest_L: int
      the biggest length of single protein.
    
    '''
    L, longest_L = read_Ngram_file(file_name, N, encode)
    protein_freq = count_frequency(L)
    print("Successfully count protein freqency!" + "(%s)" % file_name)
    
    protein_seq, protein_list = decide_seq_order(L)
    c_list = transfrom_proteinlist_into_domlist(protein_list)
    dom_seq, dom_list = decide_seq_order(c_list)
    dom_freq = count_frequency(c_list)
    print("Successfully count dom freqency!")
    
    pd_protein= produce_data_frame(protein_list, protein_freq, protein_seq,"protein")
    another_protein = pd_protein.copy()
    pd_dom= produce_data_frame(dom_list, dom_freq, dom_seq,"dom")
    data_frame = produce_proteinRank_domRank_frame(pd_protein,pd_dom,longest_L)
    print("Successfully build data frames!")
    
    return data_frame, pd_dom, another_protein, longest_L    

def write_to_excel(big, protein, dom, name):
    """Save pandas dataFrame big, protein, and dom as an excel file with the given filename
    
    ---Input
    big, protein, dom: pandas.DataFrame
        Return of info()
    
    
    
    ---Parameters
    name: string
        the name of excel file
    
    ---Output
        an excel file contains big, protein, and dom
    
    """
    writer = pd.ExcelWriter(name + '.xlsx')
    big.to_excel(writer, 'RRD')
    protein.to_excel(writer, 'protein')
    dom.to_excel(writer, 'domain')
    writer.save()

def geometric_sequence(protein, dom):
    '''give geometric sequence {Hn} and {Vm}
    
    ---Parameters
        protein, dom: pandas.DataFrame
            the output of info ()   

    ---Returns
    1. H: ndarray
        the geometric sequence of horizontal lines
        
    2. V: ndarray
        the sequence of vertical lines
      
    '''
    
    V = [0 for i in range(len(set(protein['proteinFreq'])))]
    H = [0 for i in range(len(set(dom['domFreq'])))]
    
    Vf = sorted(set(protein['proteinFreq']))
    Hf = sorted(set(dom['domFreq']))
    
    SVT = 0
    SHT = 0
    
    for i in range(len(set(protein['proteinFreq']))):
        #ref: Count how many values in a list that satisfy certain condition
        SV = sum(1 for cf in protein['proteinFreq'] if cf == Vf[i])
        SVT = SVT + SV
        V[i] = len(protein['proteinFreq']) - SVT
    #we need to add total kinds of proteins as V_1    
    V[:0] = (max(protein['proteinRank']),)    
        
    for i in range(len(set(dom['domFreq']))):
        SH = sum(1 for wf in dom['domFreq'] if wf == Hf[i])
        SHT = SHT + SH
        H[i] = len(dom['domFreq']) - SHT
    #we need to add total kinds of domain as H_1
    H[:0] = (max(dom['domRank']),)
    
    return V, H
    
def draw_RRD_plot(big, protein, dom, longest, name, V, H, need_line = 'Y', number_of_lines = 4, Color = '#ff0000', FORMAT = 'png', Path = ''):
    '''draw the RRD plot and auxiliary lines
    
    ---Input
    1. big, protein, dom, longest: pandas.DataFrame
        the return of info()
    
    2. H, V: ndarray
        the return of geometric_sequence()
    
    ---Parameters
    1. need_line: string
        If you don't want the auxiliary lines, change Y into other strings.

    2. number_of_lines: int
        the number of auxiliary lines (both horizontal and vertical lines)
    
    3. Color: colorcode
        the color of points on RRD
    
    4. FORMAT: string
        The format of your plot. Most backends support png, pdf, ps, eps and svg. 
        else: just show plot instead of saving.
    
    5. Path: file path for saving picture
        Default: save at current document
        if Path == np.nan, no figure will be saved (just show it)
        else, the figure will be saved according to Path
    
    ---Return
        coordinate: N*2 array, N = number of points
            coordinate[i][0] = x coordinate, coordinate[i][1] = y coordinate
           
    ---Output
        show or save a RRD plot
            
    '''
    fig, ax = plt.subplots()   
    if need_line == 'Y':

        Slice_number = 50 #this value decide the number of points on horizontal and vertical lines
        x_range = np.linspace(0, len(protein), Slice_number)
        y_range = np.linspace(0, len(dom), Slice_number)


        for i in range(number_of_lines):
            x_const = [V[i] for j in range(Slice_number)] #x_const =[V[i], V[i], ..., V[i]], Slice_number elements
            y_const = [H[i] for j in range(Slice_number)] #y_const =[H[i], H[i], ..., H[i]], Slice_number elements
            plt.plot(x_range, y_const) #plot y=H[i] on RRD plot
            plt.plot(x_const, y_range) #plot x=V[i] on RRD plot   

    color_list = ['#ff0000', '#CD00FF', '#ff00AB', '#ff004D', '#ff00F7', '#9100FF', '#4D00FF', '#0000FF', '#0066FF', '#00CDFF','#00FFCD', '#00FF5E','#80FF00','#EFFF00', '#FFB300']

    coordinate = []
    str_position = [i + 1 for i in range(len(big["0th_dom_rank"]))] #position starting form 1 not 0
    for i in range(longest):
        plt.plot(str_position, big[str(i) + "th_dom_rank"], 'o', markersize=3, color = Color, alpha = 0.7)
        for j in range(len(str_position)):
            if  math.isnan(big.loc[j, str(i) + "th_dom_rank"]) == False:
                coordinate.append((str_position[j], int(big.loc[j, str(i) + "th_dom_rank"])))

    #https://atmamani.github.io/cheatsheets/matplotlib/matplotlib_2/
    formatter = ticker.ScalarFormatter(useMathText = True) 
    formatter.set_scientific(True) 
    formatter.set_powerlimits((-1,1))
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    #https://stackoverflow.com/questions/34227595/how-to-change-font-size-of-the-scientific-notation-in-matplotlib
    ax.yaxis.offsetText.set_fontsize(20)
    ax.xaxis.offsetText.set_fontsize(20)

    plt.xlabel('protein', size = 25)
    plt.ylabel('domain', size = 25)
    plt.xlim([0, max(protein['proteinRank'])*11/10])
    plt.ylim([0, max(dom['domRank'])*17/15])
    plt.title(name, size = 25)
    plt.gcf().subplots_adjust(left = 0.17, bottom = 0.17)
    try:
        if Path == '':
            fig.savefig('RRD of ' + name + '.' + FORMAT, dpi = 400, format = FORMAT) 
            plt.show()
        else:
            fig.savefig(Path + 'RRD of ' + name + '.' + FORMAT, dpi = 400, format = FORMAT) 
            plt.close()
    except:
        plt.show()
    return coordinate
    
def N_dom_dist(name, big, longest, density = True, FORMAT = 'png', Path = ''):
    '''N-dom means there are N domains in one protein, it can be 1, 2, 3..., etc. 
    This function can plot their distribution
    
    ---parameters
    1. name: string
       "XXX" (your file name without filename extension)
    
    2. density: boolean, True or False
        If True, each bar will display its raw counts divided by the total number of counts.
        (y_i = y_i0 / sum_i(y_i0), where y_i0 is the raw counts of i)
        If False, each bar will display its raw counts.
    
    3. FORMAT: string
        The format of your plot. Most backends support png, pdf, ps, eps and svg. 
        else: just show plot instead of saving.
    
    4. Path: file path for saving picture
        Default: save at current document
        if Path == np.nan, no figure will be saved (just show it)
        else, the figure will be saved according to Path
    
    ---Input
        big, longest: pandas.DataFrame, int
            the output of the function info()
            where big = info()[0] and longest = info()[3]
    
    ---Output
        show or save a N-dom distribution plot
    
    '''
    N_dom = big["N_dom"]
    freq = big["proteinFreq"]
    N_dist = [0 for i in range(longest)]
    bar_x = [str(i + 1) for i in range(longest)]
    
    fig, ax = plt.subplots()
    
    for i in range(len(freq)):
        N_dist[N_dom[i] - 1] += freq[i]
    
    if density == True:
        #I don't write N_dist[N_dom[i] - 1] += freq[i]/tot because such calculation is not precise
        tot = sum(freq)
        
        for i in range(len(N_dist)):
            N_dist[i] = N_dist[i] / tot
            
        plt.bar(bar_x, N_dist, width = 1)
        plt.ylabel('$\\rho_N$ (proportion)', size = 20)
        
        
    elif density == False:
        plt.bar(bar_x, N_dist, width = 1)
        plt.ylabel('$\\rho_N$ (counts)', size = 20)
        
    else:
        print('type error: "density" should be boolean (True or False)')
    
    plt.xlabel('$N-$domain', size = 20)
    ax.tick_params(axis = 'x', labelsize = 15) 
    ax.tick_params(axis = 'y', labelsize = 15)
    #https://stackoverflow.com/questions/6774086/why-is-my-xlabel-cut-off-in-my-matplotlib-plot
    plt.gcf().subplots_adjust(left = 0.17, bottom = 0.17)
    plt.title(name, size = 20)
    try:
        if Path == '':
            fig.savefig('N-dom of ' + name + '.' + FORMAT, dpi = 300, format = FORMAT)
            plt.show()
        else:
            fig.savefig(Path + 'N-dom of ' + name + '.' + FORMAT, dpi = 300, format = FORMAT)
            plt.close()
    except:
        plt.show()

def which_plot(name, V, H, x = 'H', max_range = 50, shift = 'N', FORMAT = 'png', Path = ''):
    '''check ratio of geometric sequence {Hn} or {Vm}

    ---Parameters
    1. name: str
       "XXX" (your file name without filename extension)

    2. V, H: list or np.array
       V and H are the coordinates of the sequence {V} and {H}.
       You should get these two from 
             V, H = geometric_sequence(protein, dom)
       where geometric_sequence(protein, dom) is the function of count.py

    3. max_range: number
        the number of elements in the sequence you want to know

    4. x: 'H' or 'V'
        you can chose the sequence you want (H/V)

    5. FORMAT: string
        The format of your plot. Most backends support png, pdf, ps, eps and svg. 
        else: just show plot instead of saving.
    
    6. Path: file path for saving picture
        Default: save at current document
        if Path == np.nan, no figure will be saved (just show it)
        else, the figure will be saved according to Path
        
    ---Output
        a figure of {H_n}, or {V_m}, depend on the x = 'H' or 'V'
    
    ---Return
        mean, standard error, and shift: float
            statistical quantities of the ratio of {H} or {V}
    '''
    
    if x == 'H':
        if len(H) < max_range + 4:
            max_range = len(H) - 5
        r = np.zeros(max_range - 2)
        
        if shift == 'T':
            def r_H_shift(x_0, h):
                h = np.array(h)
                r_shift = (h[2:max_range] - x_0)/ (h[1:max_range - 1] - x_0)
                std = np.sqrt(np.mean((r_shift - r_shift.mean())**2))
                return std
            
            #To get the value minimize std of r_shift, we don't use minimize() here because 
            #there are some problems in its algorithm. Instead, we use the Brute-force search
            find_r = []
            for x_0 in range(0, int(H[0]/2)):
                find_r.append(r_H_shift(x_0, H))
            SHIFT = find_r.index(min(find_r)) + 1
            h = np.array(H)
            r = (h[2:max_range] - SHIFT)/ (h[1:max_range -1] - SHIFT)
            
            plt.ylabel('$\sigma_H(x_0)$', size = 15)
            plt.xlabel('shift $x_0$', size = 15)
            plt.text(SHIFT + 50, min(find_r) ,'$x_0=%d$' % SHIFT, fontsize = 20)
            plt.plot(find_r)
            plt.yscale('log')
            plt.show()
            

        elif shift != 'T': 
            SHIFT = 0
            for i in range(1, max_range - 1): #H[0]=H_1, H[1]=H_2
                r[i - 1] = H[i + 1]/ H[i]
        
        
        r_position = [i + 2 for i in range(len(r))] #we start from H_2
        STD = round(np.std(r), 3)
        MEAN = round(np.mean(r), 3)
        fig, ax = plt.subplots()
        ax.errorbar(r_position, r, yerr = STD) #plot errorbar 
        plt.text(max_range / 20, 0.3, '$r_H=%.3f\pm %.3f$' % (MEAN, STD), fontsize=35)        
        
        plt.title(name, size = 20)
        ax.tick_params(axis='x', labelsize=15) 
        ax.tick_params(axis='y', labelsize=15)
        plt.gcf().subplots_adjust(left = 0.17, bottom = 0.17)
        plt.xlabel('$n$ for $H_{n+1}/H_{n}$', size = 20)
        plt.ylabel('$r_H$', size = 20)
        plt.ylim([0, max(r) + 0.1])
        plt.plot(r_position, r, 'ro')        
        try:
            if Path == '':
                fig.savefig('H of ' + name + '.' + FORMAT, dpi = 400, format = FORMAT)
                plt.show()
            else:
                fig.savefig(Path + 'H of ' + name + '.' + FORMAT, dpi = 400, format = FORMAT)
                plt.close()
        except:
            plt.show()
        return MEAN, STD, SHIFT
    elif x == 'V':
        if len(V) < max_range + 4:
            max_range = len(V) - 5
        r = np.zeros(max_range - 2)
        
        if shift == 'T':
            def r_V_shift(x_0, v):
                v = np.array(v)
                r_shift = (v[2:max_range] - x_0)/ (v[1:max_range - 1] - x_0)
                std = np.sqrt(np.mean((r_shift - r_shift.mean())**2))
                return std
            
            #To get the value minimize std of r_shift, we don't use minimize() here because 
            #there are some problems in its algorithm. Instead, we use the Brute-force search
            find_r = []
            for x_0 in range(0, int(V[0]/2)):
                find_r.append(r_V_shift(x_0, V))
            SHIFT = find_r.index(min(find_r)) + 1
            v = np.array(V)
            r = (v[2:max_range] - SHIFT)/ (v[1:max_range -1] - SHIFT)
            plt.text(SHIFT + 50, min(find_r) ,'$x_0=%d$' % SHIFT, fontsize = 20)
            plt.ylabel('$\sigma_V(x_0)$', size = 15)
            plt.xlabel('shift $x_0$', size = 15)
            plt.plot(find_r)
            plt.yscale('log')
            plt.show()

        
        elif shift != 'T': 
            SHIFT = 0
            for i in range(1, max_range - 1): #V[0]=V_1, V[1]=V_2
                print(V[i], V[i+1])
                r[i - 1] = V[i + 1] / V[i]
                
        r_position = [i + 2 for i in range(len(r))] #we start from V_2
        STD = round(np.std(r), 3)
        MEAN = round(np.mean(r), 3)
        fig, ax = plt.subplots()
        ax.errorbar(r_position, r, yerr = STD) #plot errorbar
        plt.text(max_range / 20, 0.3, '$r_V=%.3f\pm %.3f$' % (MEAN, STD), fontsize=35)
        
        plt.title(name, size = 20)
        ax.tick_params(axis='x', labelsize=15) 
        ax.tick_params(axis='y', labelsize=15)
        plt.gcf().subplots_adjust(left = 0.17, bottom = 0.17)
        plt.xlabel('$m$ for $V_{m+1}/V_{m}$', size = 20)
        plt.ylabel('$r_V$', size = 20)
        plt.ylim([0, max(r) + 0.1])
        plt.plot(r_position, r, 'ro')        
        try:
            if Path == '':
                fig.savefig('V of ' + name + '.' + FORMAT, dpi = 400, format = FORMAT)
                plt.show()
            else:
                fig.savefig(Path + 'V of ' + name + '.' + FORMAT, dpi = 400, format = FORMAT)
                plt.close()
        except:
            plt.show()
        return MEAN, STD, SHIFT
    else:
        print('please chose x = \'H\' or \'V\'')

    
def FRD_plot(name, protein, dom, x_pos = 2, y_pos = 10, FORMAT = 'png', Path = ''):
    '''draw FRD plot of proteins and domains

    ---Parameters
    1. name: str
       "XXX" (your file name without filename extension)

    2. protein, dom: pd.daframe
       output of function info() or N_gram_info() in count.py
       you should get them from
       big, dom, protein, longest = info(filename, encode)

    3. x_pos, y_pos : number
       (x_position, y_position) of your formula on FRD plot

    4. FORMAT: string
        The format of your plot. Most backends support png, pdf, ps, eps and svg. 
        else: just show plot instead of saving.
    
    5. Path: file path for saving picture
        Default: save at current document
        if Path == np.nan, no figure will be saved (just show it)
        else, the figure will be saved according to Path
    
    ---Output
        save or show a figure of FRD
    
    ---Return:
        FRD_protein: dict, where
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
                
    '''
    wf = protein['proteinFreq']
    cf = dom['domFreq']
    max_wf = wf[0]
    max_cf = cf[0]

    #use MLE to get the fitting parameter, detial read: Curve_Fitting_MLE
    #-----------------------------------------
    #T = ([proteinRank], [proteinFreq])
    T = ([],[])
    for i in protein['proteinRank']:
        T[0].append(i)
    for i in wf:
        T[1].append(i)
        
    Y = Two_to_One(T)
    
    xdata = np.linspace(min(T[0]), max(T[0]), num = (max(T[0]) - min(T[0]))*10)
    
    #Estimate exponent. This action can make reduce the error of initial value guess.
    freq_M, freq_m = max(T[1]), min(T[1])
    rank_M, rank_m = max(T[0]), min(T[0])
    b_0 = np.log(freq_M / freq_m) / np.log(rank_M / rank_m)

    #fit Zipf: P(x, b)=a_Z/x^b_Z
    res_Z = minimize(L_Zipf, b_0, Y)
    b_Z = float(res_Z['x'])
    t_Z = (int(min(T[0])), int(max(T[0])), b_Z)
    a_Z = float(1 / incomplete_harmonic(t_Z))
    
    #change theo from probability density to real frequency
    L = sum(T[1]) #total number of proteins in the txt
    theo_Z = L * Zipf_law(xdata, a_Z, b_Z)
    
    #save the parameters as the return of FRD_plot()
    FRD_protein = {}
    FRD_protein['ab'] = (a_Z, b_Z)
    FRD_protein['b_jac'] = tuple(res_Z.get('jac'))
    FRD_protein['neg_L'] = res_Z.get('fun')
    FRD_protein['length'] = L
    FRD_protein['V_1'] = max(T[0])
    #-----------------------------------------
    fig, ax = plt.subplots()
    
    #plt.text(x_position, y_position)
    if (x_pos, y_pos) == (0,0):
        x_mid = 1.2
        y_min = 0.2
        plt.text(x_mid, y_min,'$b=-%.2f$'% b_Z, fontsize = 40) #write formula on the plot
    else:
        plt.text(x_pos, y_pos,'$b=-%.2f$'% b_Z, fontsize = 40) #write formula on the plot
    
        
    plt.plot(xdata, theo_Z, 'g-')
    
    plt.ylim([0.1, 10*max(max_wf, max_cf)])
    plt.plot(wf, 'ro', label = 'protein', markersize=4)
    plt.plot(cf, 'x', label = 'domain', markersize=6)
    plt.legend(loc = 'best', prop={'size': 20})
    
    plt.xlabel('Rank', size = 25)
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
            fig.savefig('FRD of ' + name + '.' + FORMAT, dpi = 400, format = FORMAT)
            plt.show()
        else:
            fig.savefig(Path + 'FRD of ' + name + '.' + FORMAT, dpi = 400, format = FORMAT)
            plt.close()
    except:
        plt.show()
    return FRD_protein

def draw_density_plot(cooridnate_x, cooridnate_y, slice_number):
    """Input cooridnate of datapoints
       draw a density diagram and slice it into slice_number pieces. 
    """
    xx = cooridnate_x
    yy = cooridnate_y
    plt.hist2d(xx, yy, slice_number, cmap = plt.cm.jet)
    plt.colorbar()
    
def read_file_generate_fake_constraint(constraint = 5, dom_num = 2, out_file =  'fake1.txt', sample_protein_num = 8000,
                            num_protein_in_fake_scrip = 15000, 
                            alpha = 1.00001, noun = False):
    """Read "roc2.txt" file, and then generate a fake script satisfying Zipfs' law. All the proteins in 
    the output script share the same lenth dom_num
    """
    CONSTRAINT = constraint
    SAMPLE_protein_NUM = sample_protein_num
    ALPHA = alpha
    NUM_protein_IN_NOV = num_protein_in_fake_scrip
    OUTPUT_FILE_NAME = out_file
    NOUN = noun
    dom_NUM = dom_num
    
    zipf_gen =  ZipfGenerator(SAMPLE_protein_NUM,ALPHA)
    f =  open("roc2.txt","r")

    world_list = []
    
    for line in f:
        line_split = line.split("\t")
        if NOUN:
            if 'N' in line_split[4]:
                world_list.append(line_split[3])
        else:
            #if len(line_split[3]) == dom_NUM:
                world_list.append(line_split[3])

    f.close()
    
    for item in world_list:
        if item == " ":
            world_list.remove(item)
    #######################################
    ##########produce fake proteins###########
    
    tmp_list = []
    for item in world_list:
        for e in list(item):
            if e not in tmp_list:
                tmp_list.append(e)
    dom_count_dic = {}
    for c in tmp_list:
        dom_count_dic[c] = 0
    

        
    
    list_2 = []
    tmp = ''
    for i in range(SAMPLE_protein_NUM):
        for j in range(dom_num):
            c = random.choice(tmp_list)
            dom_count_dic[c] += 1
            if dom_count_dic[c] >= CONSTRAINT:
                tmp_list.remove(c)
            tmp = tmp + c
        list_2.append(tmp)
        tmp = ''
    
    world_list = list_2[:]

    print("proteins in corpus: " ,len(world_list))
    
    
    #######################################


    print("A corpus is successfully loaded.")
    
    random.shuffle(world_list)
    small_world_list = world_list[:]
    target_string_list = []

    for i in range(NUM_protein_IN_NOV):
        num = zipf_gen.next()
        w = small_world_list[num]
        target_string_list.append(w+" ")
        
    f2 = open(OUTPUT_FILE_NAME , 'w')

    protein_count = 0
    for item in target_string_list:
        if protein_count < 20:
            f2.write(item)
            protein_count += 1
        else:
            protein_count = 0
            f2.write(item+"\n")
    f2.close()
    print("A fake script is successfully created !")
    print("--------------------")
    return None   


def read_file_generate_fake(dom_num = 2, out_file =  'fake1.txt', sample_protein_num = 8000,
                            num_protein_in_fake_scrip = 15000, 
                            alpha = 1.00001, noun = False):
    """Read "roc2.txt" file, and then generate a fake script satisfying Zipfs' law. All the proteins in 
    the output script share the same lenth dom_num
    """
    SAMPLE_protein_NUM = sample_protein_num
    ALPHA = alpha
    NUM_protein_IN_NOV = num_protein_in_fake_scrip
    OUTPUT_FILE_NAME = out_file
    NOUN = noun
    dom_NUM = dom_num
    
    zipf_gen =  ZipfGenerator(SAMPLE_protein_NUM,ALPHA)
    f =  open("roc2.txt","r")

    world_list = []
    
    for line in f:
        line_split = line.split("\t")
        if NOUN:
            if 'N' in line_split[4]:
                world_list.append(line_split[3])
        else:
            #if len(line_split[3]) == dom_NUM:
                world_list.append(line_split[3])

    f.close()
    
    for item in world_list:
        if item == " ":
            world_list.remove(item)
    #######################################
    ###these codes are optional 
    
    tmp_list = []
    for item in world_list:
        for e in list(item):
            tmp_list.append(e)
    random.shuffle(tmp_list)
    list_2 = []
    tmp = ''
    for e in tmp_list:
        tmp = tmp + e
        if len(tmp) == dom_num:
            list_2.append(tmp)
            tmp = ''
    
    world_list = list_2

    print("proteins in a corpus: " ,len(world_list))
    
    
    #######################################


    print("A corpus is successfully loaded.")
    
    random.shuffle(world_list)
    small_world_list = world_list[-SAMPLE_protein_NUM:]
    target_string_list = []

    for i in range(NUM_protein_IN_NOV):
        num = zipf_gen.next()
        w = small_world_list[num]
        target_string_list.append(w+" ")
        
    f2 = open(OUTPUT_FILE_NAME , 'w')

    protein_count = 0
    for item in target_string_list:
        if protein_count < 20:
            f2.write(item)
            protein_count += 1
        else:
            protein_count = 0
            f2.write(item+"\n")
    f2.close()
    print("A fake script is successfully created !")
    print("--------------------")
    return None