# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 17:02:47 2016

@author: shan, gmking

This module is used to construct a dataframe with all statistical information we need.
The core function of this module is info(file_name, encode = "UTF-8")


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
    return a list of the words of text in the file; ignore punctuations.
    also returns the longest word length in the file.
    
    paras:
    --------
    file_name : string
      XXX.txt. We suggest you using the form that set 
      name = 'XXX' 
      and 
      filename = name + '.txt'.

    encode : encoding of your txt
    """
    punctuation_set = set(u'''_—＄％＃＆:#$&!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒
    ﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠
    々‖•·ˇˉ－―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻
    ︽︿﹁﹃﹙﹛﹝（｛“‘-—_…''')
    num = 0
    word_list = []
    with open(filename, "r", encoding = encode) as file:
        for line in file:
            l = line.split()
            new_word = ''
            for word in l:
                for c in word:
                    if c not in punctuation_set:
                        new_word = new_word + c
                    if c in punctuation_set and len(new_word) != 0:
                        word_list.append(new_word)
                        new_word = ''
                if len(new_word) != 0: 
                    if len(new_word) > num:
                        num = len(new_word) #max number of syllagrams in a word
                    word_list.append(new_word)
                    new_word = ''
                    
    if '\ufeff' in word_list:
        word_list.remove('\ufeff')
    
    print("read file successfully!")
    return word_list, num

def read_Ngram_file(filename, N, encode = 'UTF-8'):
    """
    Read the text file with the given filename;    return a list of the words of text in the file; ignore punctuations.
    
    paras:
    --------
    file_name : string
      XXX.txt. We suggest you using the form that set 
      name = 'XXX' 
      and 
      filename = name + '.txt'.
        
    N: int
      "N"-gram. 
      For example : a string, ABCDEFG (In Chinese, you don't know what's the 'words' of a string)
      in 2-gram = [AB, CD, EF, G];
      in 3-gram = [ABC, DEF, G];
      in 4-gram = [ABCD, EFG]
      
      two word compose a txt 'ABCDE EFGHI' (This case happended in English corpus)
      in 2-gram = [AB, CD, E, EF, GH, I];
      in 3-gram = [ABC, DE, EFG, HI];
      in 4-gram = [ABCD, E, EFGH, I]
    
    encode : encoding of your txt
    """
    punctuation_set = set(u'''_—＄％＃＆:#$&!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒
    ﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠
    々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻
    ︽︿﹁﹃﹙﹛﹝（｛“‘-—_…''')

    word_list = []
    with open(filename, "r", encoding = encode) as file:
        for line in file:
            new_word = ''
            for word in line:
                if word not in punctuation_set:
                    new_word += word
                if len(new_word) != 0 and (word in punctuation_set or len(new_word) == N):
                    word_list.append(new_word)
                    new_word = ''
    
    if '\ufeff' in word_list:
        word_list.remove('\ufeff')
                    
    print("read file successfully!")
    return word_list, N


def count_frequency(word_list):
    """
    Input: 
        word_list: list
            a list containing words or syllagrams
    Return: 
        D: set
            a dictionary mapping words to frequency.
    """
    D = {}
    for new_word in word_list:
        if new_word in D:
            D[new_word] = D[new_word] + 1
        else:
            D[new_word] = 1
    return D   


def decide_seq_order(word_list):
    """
    Input:
        word_list: list
            a list containing words or syllagrams
    Return: 
        D: set
            a dictionary mapping each word to its sequential number, which is decided by the order it 
            first appears in the word_list.
        another_list: list
            a list containg non-repetitive words, each in the order it first appears in word_list.
    """
    D = {}
    another_list = []
    for word in word_list:
        if word not in another_list:
            another_list.append(word)
    for num in range(len(another_list)):
        D[another_list[num]] = num + 1
    
    return D, another_list


def transfrom_wordlist_into_syllist(word_list):
    """Divide each words in the word_list into syllagrams, order reserved.
    Input: a list containing words
    Return: a list containg syl 
    """
    syl_list = []
    for word in word_list:
        syl_list.extend(list(word))
        
    return syl_list


def produce_data_frame(word_list, word_freq, word_seq, varibleTitle):
    word_list = list(set(word_list))
    data = {}
    word_seq_list = []
    word_freq_list = []
    
    for word in word_list:
        word_freq_list.append(word_freq[word])
        word_seq_list.append(word_seq[word])
    
    first = varibleTitle 
    second = varibleTitle + "SeqOrder"
    third = varibleTitle + "Freq"
    forth = varibleTitle + "Rank"
    
    data[first] = word_list
    data[second] = word_seq_list
    data[third] = word_freq_list  
    
    dataFrame = pd.DataFrame(data)
    dataFrame = dataFrame.sort_values([third, second],ascending = [False,True])
    rank = np.array(list(range(1,len(dataFrame)+1))) 
    dataFrame[forth] = rank
    column_list = [first, third, forth, second]
    dataFrame = dataFrame[column_list]
    dataFrame = dataFrame.reset_index(drop=True)
    return dataFrame


def produce_wordRank_sylRank_frame(pd_word,pd_syl,longest):
    
    D = {}
    
    syl_array = pd_syl["syl"]
    syl_rank = {}
    
    for i in range(len(pd_syl)):
        syl_rank[syl_array[i]] = i + 1 
    
    for i in range(longest):
        D[i] = []
    
    word_array = pd_word["word"]
    N_syllagram = [] #count how many syllagram in a word
    
    for word in word_array:
        N_syllagram.append(len(word))
        
        for i in range(len(word)):
            D[i].append(int(syl_rank[word[i]]))
        
        if len(word) < longest:
            for j in range(len(word),longest):
                D[j].append(np.nan)
    
    pd_word["N_syl"] = np.array(N_syllagram)
    
    for k in range(longest):
        feature = str(k) + "th" + "_syl_rank"
        pd_word[feature] = np.array(D[k])
    
    return pd_word  


def info(file_name, encode = "UTF-8"):
    '''This is the main program.
        
    paras:
    --------
    file_name : string
      XXX.txt. We suggest you using the form that set 
      name = 'XXX' 
      and 
      filename = name + '.txt'.
    
    encode : encoding of your txt
    
    
    return:
    --------
    data_frame: pd.dataframe
      a big data frame contain the information of words and its compositition
    pd_syl: pd.dataframe
      a data frame contain the information of syllagrams
    another_word: pd.dataframe
      a data frame contain the information of words
    longest_L: int
      the biggest length of single word.
    
    '''
    
    L, longest_L = read_file(file_name,encode)
    word_freq = count_frequency(L)
    print("Successfully count word freqency!" + "(%s)" % file_name)
    
    word_seq, word_list = decide_seq_order(L)
    c_list = transfrom_wordlist_into_syllist(L)
    syl_seq, syl_list = decide_seq_order(c_list)
    syl_freq = count_frequency(c_list)
    print("Successfully count syl freqency!")
    
    pd_word= produce_data_frame(word_list, word_freq, word_seq,"word")
    another_word = pd_word.copy()
    pd_syl= produce_data_frame(syl_list, syl_freq, syl_seq,"syl")
    data_frame = produce_wordRank_sylRank_frame(pd_word,pd_syl,longest_L)
    print("Successfully build data frames!")
    
    return data_frame, pd_syl, another_word, longest_L

def N_gram_info(file_name, N, encode = "UTF-8"):
    '''This is only used to analysis N-gram words.
        
    paras:
    --------
    file_name : string
      XXX.txt. We suggest you using the form that set 
      name = 'XXX' 
      and 
      filename = name + '.txt'.
        
    N: int
      "N"-gram. 
      For example : a string, ABCDEFG (In Chinese, you don't know what's the 'words' of a string)
      in 2-gram = [AB, CD, EF, G];
      in 3-gram = [ABC, DEF, G];
      in 4-gram = [ABCD, EFG]
      
      two word compose a txt 'ABCDE EFGHI' (This case happended in English corpus)
      in 2-gram = [AB, CD, E, EF, GH, I];
      in 3-gram = [ABC, DE, EFG, HI];
      in 4-gram = [ABCD, E, EFGH, I]
    
    encode : encoding of your txt
    
    
    return:
    --------
    data_frame: pd.dataframe
      a big data frame contain the information of words and its compositition
    pd_syl: pd.dataframe
      a data frame contain the information of syllagrams
    another_word: pd.dataframe
      a data frame contain the information of words
    longest_L: int
      the biggest length of single word.
    
    '''
    L, longest_L = read_Ngram_file(file_name, N, encode)
    word_freq = count_frequency(L)
    print("Successfully count word freqency!" + "(%s)" % file_name)
    
    word_seq, word_list = decide_seq_order(L)
    c_list = transfrom_wordlist_into_syllist(L)
    syl_seq, syl_list = decide_seq_order(c_list)
    syl_freq = count_frequency(c_list)
    print("Successfully count syl freqency!")
    
    pd_word= produce_data_frame(word_list, word_freq, word_seq,"word")
    another_word = pd_word.copy()
    pd_syl= produce_data_frame(syl_list, syl_freq, syl_seq,"syl")
    data_frame = produce_wordRank_sylRank_frame(pd_word,pd_syl,longest_L)
    print("Successfully build data frames!")
    
    return data_frame, pd_syl, another_word, longest_L


def write_to_excel(big, word, syl, name):
    """Write pandas dataFrame big, word, syl to an excel file with the given filename
    """
    writer = pd.ExcelWriter(name + '.xlsx')
    big.to_excel(writer,'RRD')
    word.to_excel(writer,'word')
    syl.to_excel(writer,'syllagram')
    writer.save()


def geometric_sequence(word, syl):
    '''give geometric sequence {Hn} and {Vn}
    
    paras:
    ---
    word, syl: pandas.DataFrame
        the output of info    
    
    returns:
    ---
    H: ndarray
        the geometric sequence of horizontal lines
    V: ndarray
        the sequence of vertical lines
      
    '''
    
    V = [0 for i in range(len(set(word['wordFreq'])))]
    H = [0 for i in range(len(set(syl['sylFreq'])))]
    
    Vf = sorted(set(word['wordFreq']))
    Hf = sorted(set(syl['sylFreq']))
    
    SVT = 0
    SHT = 0
    
    for i in range(len(set(word['wordFreq']))):
        #ref: Count how many values in a list that satisfy certain condition
        SV = sum(1 for cf in word['wordFreq'] if cf == Vf[i])
        SVT = SVT + SV
        V[i] = len(word['wordFreq']) - SVT + 1
    V[:0] = (max(word['wordRank']),)    
        
    for i in range(len(set(syl['sylFreq']))):
        SH = sum(1 for wf in syl['sylFreq'] if wf == Hf[i])
        SHT = SHT + SH
        H[i] = len(syl['sylFreq']) - SHT + 1
    H[:0] = (max(syl['sylRank']),)
    
    return V, H
    

def draw_RRD_plot(big, word, syl, longest, name, V, H, need_line = 'Y', number_of_lines = 4, Color = '#ff0000', FORMAT = 'png', Path = ''):
    '''draw the RRD plot and auxiliary lines
    
    Controllable parameters:
    --- 
    need_line: string
        If you don't want the auxiliary lines, change Y into other thing.

    number_of_lines: number
        How many auxiliary lines you need ? (both horizontal and vertical lines)
    Color: colorcode
    FORMAT: string
        The format of your RRD plot. Most backends support png, pdf, ps, eps and svg. 
        else: just show plot instead of saving.
    Path: file path for saving picture
        Default: save at current document
    
    Fixed parameters:
    ---(please don't change them)
    big, word, syl, longest: pandas.DataFrame
        the output of the function info()
    H, V: ndarray
        the output of the function check_const_ratio
           
    output:
        1. show or save a RRD plot
        2. coordinate: N*2 array, N = number of points
                coordinate[i][0] = x coordinate, coordinate[i][1] = y coordinate
            
    '''
    fig, ax = plt.subplots()   
    if need_line == 'Y':

        Slice_number = 50 #this value decide the number of points on horizontal and vertical lines
        x_range = np.linspace(0, len(word), Slice_number)
        y_range = np.linspace(0, len(syl), Slice_number)


        for i in range(number_of_lines):
            x_const = [V[i] for j in range(Slice_number)]#x_const =[V[i], V[i], ..., V[i]], Slice_number elements
            y_const = [H[i] for j in range(Slice_number)] #y_const =[H[i], H[i], ..., H[i]], Slice_number elements
            plt.plot(x_range, y_const) #plot y=H[i] on RRD plot
            plt.plot(x_const, y_range) #plot x=V[i] on RRD plot   

    color_list = ['#ff0000', '#CD00FF', '#ff00AB', '#ff004D', '#ff00F7', '#9100FF', '#4D00FF', '#0000FF', '#0066FF', '#00CDFF','#00FFCD', '#00FF5E','#80FF00','#EFFF00', '#FFB300']

    coordinate = []
    str_position = [i + 1 for i in range(len(big["0th_syl_rank"]))] #position starting form 1 not 0
    for i in range(longest):
        plt.plot(str_position, big[str(i) + "th_syl_rank"], 'o', markersize=3, color = Color, alpha = 0.7)
        for j in range(len(str_position)):
            if  math.isnan(big.loc[j, str(i) + "th_syl_rank"]) == False:
                coordinate.append((str_position[j], int(big.loc[j, str(i) + "th_syl_rank"])))

    #https://atmamani.github.io/cheatsheets/matplotlib/matplotlib_2/
    formatter = ticker.ScalarFormatter(useMathText = True) 
    formatter.set_scientific(True) 
    formatter.set_powerlimits((-1,1))
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    #https://stackoverflow.com/questions/34227595/how-to-change-font-size-of-the-scientific-notation-in-matplotlib
    ax.yaxis.offsetText.set_fontsize(15)
    ax.xaxis.offsetText.set_fontsize(15)

    plt.xlabel('word', size = 15)
    plt.ylabel('syllagram', size = 15)
    plt.xlim([0, max(word['wordRank'])*11/10])
    plt.ylim([0, max(syl['sylRank'])*17/15])
    plt.title(name, size = 25)
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
    
def N_syl_dist(name, big, longest, FORMAT = 'png', Path = ''):
    '''N-syl means there are N syllagrams in one word, it can be 1, 2, 3..., etc. This function can print their distribution
    
    Controllable parameters:
    --- 
    name: str
       "XXX" (your file name without filename extension)
    FORMAT: string
        The format of your N-syl plot. Most backends support png, pdf, ps, eps and svg.
    
    Fixed parameters:
    ---(please don't change them)
    big, longest: pandas.DataFrame, int
        the output of the function info()
    
    Output:
    ---
        a N-syl distribution plot
    
    '''
    N_syl = big["N_syl"]
    fig, ax = plt.subplots()
    plt.hist(N_syl, bins = longest, density = True)
    
    plt.xlabel('$N-syllagram$', size = 20)
    plt.ylabel('$\\rho(N)$', size = 20)
    ax.tick_params(axis='x', labelsize=15) 
    ax.tick_params(axis='y', labelsize=15)
    #https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.ticklabel_format.html
    ax.ticklabel_format(axis='x', style='sci',scilimits=(0,3))
    #https://stackoverflow.com/questions/6774086/why-is-my-xlabel-cut-off-in-my-matplotlib-plot
    plt.gcf().subplots_adjust(left = 0.17, bottom = 0.17)
    plt.title(name, size = 20)
    try:
        if Path == '':
            fig.savefig('N-syl of ' + name + '.' + FORMAT, dpi = 300, format = FORMAT)
            plt.show()
        else:
            fig.savefig(Path + 'N-syl of ' + name + '.' + FORMAT, dpi = 300, format = FORMAT)
            plt.close()
    except:
        plt.show()

def which_plot(name, V, H, x = 'H', max_range = 50, shift = 'N', FORMAT = 'png', Path = ''):
    '''check ratio of geometric sequence {Hn} or {Vn}

       parameters:
    1. name: str
       "XXX" (your file name without filename extension)

    2. V, H: list or np.array
       V and H are the coordinates of the sequence {V} and {H}.
       You should get these two from 
             V, H = geometric_sequence(word, syl)
       where geometric_sequence(word, syl) is the function of count.py

    3. max_range: number
        the number of elements in the sequence you want to know

    4. x: 'H' or 'V'
        you can chose the sequence you want (H/V)

    5. FORMAT: png, pdf, ps, eps and svg
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
        plt.xlabel('$\ell$ for $H_{\ell+1}/H_{\ell}$', size = 20)
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
        plt.xlabel('$m$ for $V_{m+1}/H_{m}$', size = 20)
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

    
def FRD_plot(name, word, syl, x_pos = 2, y_pos = 10, FORMAT = 'png', Path = ''):
    '''draw FRD plot of words and syllagrams

       parameters:
    1. name: str
       "XXX" (your file name without filename extension)

    2. word, syl: pd.daframe
       output of function info() or N_gram_info() in count.py
       you should get them from
       big, syl, word, longest = info(filename, encode)

    3. x_pos, y_pos : number
       (x_position, y_position) of your formula on FRD plot

    4. FORMAT: string
        The format of your RRD plot. Most backends support png, pdf, ps, eps and svg. 
        else: just show plot instead of saving.
    
    5. Path: file path for saving picture
        Default: save at current document
    
        output:
    (C, s): normalized coeffecient and exponent for P(x)=Cx^s
    '''
    wf = word['wordFreq']
    cf = syl['sylFreq']
    max_wf = wf[0]
    max_cf = cf[0]

    #use MLE to get the fitting parameter, detial read: Curve_Fitting_MLE
    #-----------------------------------------
    T = ([],[])
    for i in word['wordRank']:
        T[0].append(i)
    for i in wf:
        T[1].append(i)
    #T = ([wordRank], [wordFreq])
    Y = Two_to_One(T)
    res = minimize(L_Zipf, 1.2, Y, method = 'SLSQP')
    s = res['x']
    t = [int(min(T[0])), int(max(T[0])), s]
    C = 1 / incomplete_harmonic(t)
    fig, ax = plt.subplots()
    plt.xlabel('rank', size = 20)
    plt.ylabel('frequency', size = 20)
    plt.title(name, fontsize = 25)

    xdata = np.linspace(min(T[0]), max(T[0]), num = (max(T[0]) - min(T[0]))*10)
    theo = Zipf_law(xdata, s, C) #Notice theo is normalized, i.e, the probability density
    N = sum(T[1])
    theo = [N * i for i in theo] #change theo from probability density to real frequency
    
    #plt.text(x_position, y_position)
    if (x_pos, y_pos) == (0,0):
        x_mid = 1.2
        y_min = 0.2
        plt.text(x_mid, y_min,'$%.3fx^{-%.2f}$'%(C, s), fontsize=40) #write formula on the plot
    else:
        plt.text(x_pos, y_pos,'$%.3fx^{-%.2f}$'%(C, s), fontsize=40) #write formula on the plot
    
        
    plt.plot(xdata, theo, 'g-')
    #-----------------------------------------
    plt.ylim([0.1, 10*max(max_wf, max_cf)])
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(wf, 'ro', label = 'word', markersize=4)
    plt.plot(cf, 'x', label = 'syl', markersize=6)
    plt.legend(loc = 'best', prop={'size': 20})
    
    #https://atmamani.github.io/cheatsheets/matplotlib/matplotlib_2/
    formatter = ticker.ScalarFormatter(useMathText=True) 
    formatter.set_scientific(True) 
    formatter.set_powerlimits((-1,1))
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter) 
    
    #https://stackoverflow.com/questions/34227595/how-to-change-font-size-of-the-scientific-notation-in-matplotlib
    ax.xaxis.offsetText.set_fontsize(15)
    ax.yaxis.offsetText.set_fontsize(15) 
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    
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
    return (C, s)

def draw_density_plot(cooridnate_x, cooridnate_y, slice_number):
    """input cooridnate of datapoints
       draw a density diagram and slice it into slice_number pieces. 
    """
    xx = cooridnate_x
    yy = cooridnate_y
    plt.hist2d(xx, yy, slice_number, cmap = plt.cm.jet)
    plt.colorbar()
    
def read_file_generate_fake_constraint(constraint = 5, syl_num = 2, out_file =  'fake1.txt', sample_word_num = 8000,
                            num_word_in_fake_scrip = 15000, 
                            alpha = 1.00001, noun = False):
    """Read "roc2.txt" file, and then generate a fake script satisfying Zipfs' law. All the words in 
    the output script share the same lenth syl_num
    """
    CONSTRAINT = constraint
    SAMPLE_WORD_NUM = sample_word_num
    ALPHA = alpha
    NUM_WORD_IN_NOV = num_word_in_fake_scrip
    OUTPUT_FILE_NAME = out_file
    NOUN = noun
    SYL_NUM = syl_num
    
    zipf_gen =  ZipfGenerator(SAMPLE_WORD_NUM,ALPHA)
    f =  open("roc2.txt","r")

    world_list = []
    
    for line in f:
        line_split = line.split("\t")
        if NOUN:
            if 'N' in line_split[4]:
                world_list.append(line_split[3])
        else:
            #if len(line_split[3]) == SYL_NUM:
                world_list.append(line_split[3])

    f.close()
    
    for item in world_list:
        if item == " ":
            world_list.remove(item)
    #######################################
    ##########produce fake words###########
    
    tmp_list = []
    for item in world_list:
        for e in list(item):
            if e not in tmp_list:
                tmp_list.append(e)
    syl_count_dic = {}
    for c in tmp_list:
        syl_count_dic[c] = 0
    

        
    
    list_2 = []
    tmp = ''
    for i in range(SAMPLE_WORD_NUM):
        for j in range(syl_num):
            c = random.choice(tmp_list)
            syl_count_dic[c] += 1
            if syl_count_dic[c] >= CONSTRAINT:
                tmp_list.remove(c)
            tmp = tmp + c
        list_2.append(tmp)
        tmp = ''
    
    world_list = list_2[:]

    print("words in corpus: " ,len(world_list))
    
    
    #######################################


    print("A corpus is successfully loaded.")
    
    random.shuffle(world_list)
    small_world_list = world_list[:]
    target_string_list = []

    for i in range(NUM_WORD_IN_NOV):
        num = zipf_gen.next()
        w = small_world_list[num]
        target_string_list.append(w+" ")
        
    f2 = open(OUTPUT_FILE_NAME , 'w')

    word_count = 0
    for item in target_string_list:
        if word_count < 20:
            f2.write(item)
            word_count += 1
        else:
            word_count = 0
            f2.write(item+"\n")
    f2.close()
    print("A fake script is successfully created !")
    print("--------------------")
    return None   


def read_file_generate_fake(syl_num = 2, out_file =  'fake1.txt', sample_word_num = 8000,
                            num_word_in_fake_scrip = 15000, 
                            alpha = 1.00001, noun = False):
    """Read "roc2.txt" file, and then generate a fake script satisfying Zipfs' law. All the words in 
    the output script share the same lenth syl_num
    """
    SAMPLE_WORD_NUM = sample_word_num
    ALPHA = alpha
    NUM_WORD_IN_NOV = num_word_in_fake_scrip
    OUTPUT_FILE_NAME = out_file
    NOUN = noun
    SYL_NUM = syl_num
    
    zipf_gen =  ZipfGenerator(SAMPLE_WORD_NUM,ALPHA)
    f =  open("roc2.txt","r")

    world_list = []
    
    for line in f:
        line_split = line.split("\t")
        if NOUN:
            if 'N' in line_split[4]:
                world_list.append(line_split[3])
        else:
            #if len(line_split[3]) == SYL_NUM:
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
        if len(tmp) == syl_num:
            list_2.append(tmp)
            tmp = ''
    
    world_list = list_2

    print("words in a corpus: " ,len(world_list))
    
    
    #######################################


    print("A corpus is successfully loaded.")
    
    random.shuffle(world_list)
    small_world_list = world_list[-SAMPLE_WORD_NUM:]
    target_string_list = []

    for i in range(NUM_WORD_IN_NOV):
        num = zipf_gen.next()
        w = small_world_list[num]
        target_string_list.append(w+" ")
        
    f2 = open(OUTPUT_FILE_NAME , 'w')

    word_count = 0
    for item in target_string_list:
        if word_count < 20:
            f2.write(item)
            word_count += 1
        else:
            word_count = 0
            f2.write(item+"\n")
    f2.close()
    print("A fake script is successfully created !")
    print("--------------------")
    return None