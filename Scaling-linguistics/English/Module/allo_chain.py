# -*- coding: utf-8 -*-
"""
@author  gmking

This module is used to calculate the allocations and chains.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit

def count_allo(pdframe1, pdframe2, feature1 = "word", feature2 = "syl", feature3 = "sylFreq"):
    '''count the allocations of syllagrams and the chains of words
    
    input:
        pdframe1, pdframe2 : class pandas.DataFrame.
        This two args are from module > count.py > info(file_name, encode = "UTF-8")
        They would be word and syl (See Run_case_by_case.ipynb)
    
    output:
        add a frame "#allocations" (numbers of allocations of syls) in pdframe2
        add a frame "#chains" (numbers of chains of words) in pdframe1
    
    return:
        none
        
    '''
    
    word_array = pdframe1[feature1] #ex: word_array=['apple','coffee','elephant']
    syl_array = pdframe2[feature2] #ex: syl_array=['ap', 'ple', 'cof', 'fee', 'e', 'le', 'phant']
    
    #First, we calculate allocations
    
    allocation = {}
    for c in syl_array: 
        allocation[c] = 0
        for w in word_array:
            t = w.split('-')
            if c in set(t): #ex: 'A' in 'AB', but 'A' not in 'BC'
                allocation[c] += 1

    syl_num_allocations_array = np.array([], dtype = 'int16' )
    
    for i in range(len(pdframe2)):
        syl_num_allocations_array = np.append(syl_num_allocations_array, allocation[syl_array[i]])
    
    #add a frame "#allocations" (numbers of allocations of syls) to syl
    pdframe2['#allocations'] = syl_num_allocations_array 
        
    #Second, we use allocation to calculate chains
    
    
    chain = {}
    for w in word_array:
        chain[w] = 0
        t = w.split('-')
        for c in set(t):
            #If we don't use set(w) here, the chains will be overcount. 
            #ex: chain('AA') = allocation('A') but not 2*allocation('A')
            chain[w] += allocation[c]
    
    chain_num_array = np.array([], dtype = 'int16')
    
    for i in range(len(pdframe1)):
        chain_num_array = np.append(chain_num_array , chain[word_array[i]])
    
    #add a frame "#chains" (numbers of chains of words) to word
    pdframe1['#chains'] = chain_num_array 

    return None

def Allo_plot(name, syl, x_pos = 0, y_pos = 0, FORMAT = 'png', Path = ''):
    '''draw FRD plot of words and syllagrams

       parameters:
    1. name: str
       "XXX" (your file name without filename extension)

    2. syl: pd.daframe
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
    figure of allocation distribution
    '''
    Syl = syl.sort_values(by = '#allocations', ascending=False)
    reSyl = Syl.reset_index()

    #use OLS to get the fitting parameter
    #-----------------------------------------
    def allo(y, a, b):
        return (a * np.log(y) + b) ** 2

    popt, pcov = curve_fit(allo, syl['sylRank'], reSyl['#allocations'])
    #popt is the optimal values for the parameters (a,b)
    theo = allo(syl['sylRank'], *popt)
    fig, ax = plt.subplots()
    plt.plot(syl['sylRank'], theo, 'g--')
    plt.plot(reSyl['#allocations'], 'ro', label = 'syl', markersize = 4)
    
    #the following code deal with significant figures of fitting parameters
    #the tutor of significant figures: https://www.usna.edu/ChemDept/_files/documents/manual/apdxB.pdf
    #-----------------------------------------
    allo_dig = len(str(max(reSyl['#allocations'])))
    yp_dig = len(str(max(syl['sylRank']))) #ln(y') will have yp_dig +1 digits (yp_dig significant figures)
    a_dig = min(allo_dig, yp_dig +1) #significant figures of parameter a
    b_dig = allo_dig #significant figures of parameter b
    
    # the fomat string is #.?g, where ? = significant figures
    # detail of the fomat string: https://bugs.python.org/issue32790
    # https://docs.python.org/3/tutorial/floatingpoint.html
    A = format(abs(popt[0]), '#.%dg' % a_dig)  # give a_dig significant digits
    B = format(popt[1], '#.%dg' % b_dig)  # give b_dig significant digits
    if 'e' in A: #make scientific notation more beautiful
        A = A.split('e')[0] + '\\times 10^{' + str(int(A.split('e')[1])) + '}'
    if A[-1] == '.':
        A = A[:-1]
    if 'e' in B: #make scientific notation more beautiful
        B = B.split('e')[0] + '\\times 10^{' + str(int(B.split('e')[1])) + '}'
    if B[-1] == '.':
        B = B[:-1]
        
    #a perfect solution to text wrap!!
    #https://stackoverflow.com/questions/2660319/putting-newline-in-matplotlib-label-with-tex-in-python
    parameters = (r"$\alpha=%s$"
                  "\n"
                 r"$\beta=%s$") % (A, B)    
    
    a = 1.5  #auto positioning for m = min(syl['sylRank']) = 1 always
    b = 2   #auto positioning for M = max(syl['sylRank'])
    xmid = max(syl['sylRank'])**(b/(a+b))  #exp([a*log(m)+b*log(M)]/[a+b]) = m^(a/[a+b]) * M^(b/[a+b])    
    ytop = max(reSyl['#allocations'])*5/7
    
    if x_pos != 0 and y_pos != 0:
        plt.text(x_pos, y_pos, parameters, fontsize=30)
    elif x_pos != 0 and y_pos == 0:
        plt.text(x_pos, ytop, parameters, fontsize=30)
    elif x_pos == 0 and y_pos != 0:
        plt.text(xmid, y_pos, parameters, fontsize=30)
    else:
        plt.text(xmid, ytop, parameters, fontsize=30)
    #-----------------------------------------
    plt.xlabel('Rank of syllagram($y\prime$)', size = 20)
    plt.xscale('log')
    plt.ylabel('$Allo(y\prime)$', size = 20)
    ax.tick_params(axis='x', labelsize=15) 
    ax.tick_params(axis='y', labelsize=15)
    #https://stackoverflow.com/questions/6774086/why-is-my-xlabel-cut-off-in-my-matplotlib-plot
    plt.gcf().subplots_adjust(left = 0.17, bottom = 0.17)
    plt.title(name, fontsize = 20)
    try:
        if Path == '':
            fig.savefig('allocation_' + name + '.' + FORMAT, dpi = 500, format = FORMAT)
            plt.show()
        else:
            fig.savefig(Path + 'allocation_' + name + '.' + FORMAT, dpi = 500, format = FORMAT)
            plt.close()
    except:
        plt.show()
        
def Chain_plot(name, word, x_pos = 0, y_pos = 0, FORMAT = 'png', Path = ''):
    '''draw allocation-rank plot 

       parameters:
    1. name: str
       "XXX" (your file name without filename extension)

    2. word: pd.daframe
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
    figure of chain distribution
    '''
    Word = word.sort_values(by='#chains', ascending=False)
    reWord = Word.reset_index()

    #use OLS to get the fitting parameter
    #-----------------------------------------
    def chain(x, a, b):
        return (a * np.log(x) + b)

    popt, pcov = curve_fit(chain, word['wordRank'], reWord['#chains'])
    #popt is the optimal values for the parameters (a,b)
    theo = chain(word['wordRank'], *popt)
    fig, ax = plt.subplots()
    plt.plot(word['wordRank'], theo, 'g--')
    plt.plot(reWord['#chains'], 'ro', label = 'word', markersize = 4)


    
    #the following code deal with significant figures of fitting parameters
    #the tutor of significant figures: https://www.usna.edu/ChemDept/_files/documents/manual/apdxB.pdf
    #-----------------------------------------
    chain_dig = len(str(max(reWord['#chains'])))
    xp_dig = len(str(max(word['wordRank']))) #ln(x') will have xp_dig +1 digits (xp_dig significant figures)
    a_dig = min(chain_dig, xp_dig +1) #significant figures of parameter a
    b_dig = chain_dig #significant figures of parameter b
    
    # the fomat string is #.?g, where ? = significant figures
    # detail of the fomat string: https://bugs.python.org/issue32790
    # https://docs.python.org/3/tutorial/floatingpoint.html
    A = format(abs(popt[0]), '#.%dg' % a_dig)  # give a_dig significant digits
    B = format(popt[1], '#.%dg' % b_dig)  # give b_dig significant digits
    if 'e' in A: #make scientific notation more beautiful
        A = A.split('e')[0] + '\\times 10^{' + str(int(A.split('e')[1])) + '}'
    if A[-1] == '.':
        A = A[:-1]
    if 'e' in B: #make scientific notation more beautiful
        B = B.split('e')[0] + '\\times 10^{' + str(int(B.split('e')[1])) + '}'
    if B[-1] == '.':
        B = B[:-1]    
    
    #a perfect solution to text wrap!!
    #https://stackoverflow.com/questions/2660319/putting-newline-in-matplotlib-label-with-tex-in-python
    parameters = (r"$\gamma=%s$"
                  "\n"
                 r"$\omega=%s$") % (A, B)    
    
    a = 1.5  #auto positioning for m = min(syl['sylRank']) = 1 always
    b = 2   #auto positioning for M = max(syl['sylRank'])
    xmid = max(word['wordRank']) ** (b/(a+b))  #exp([a*log(m)+b*log(M)]/[a+b]) = m^(a/[a+b]) * M^(b/[a+b])
    ytop = max(reWord['#chains'])*5/7
    
    if x_pos != 0 and y_pos != 0:
        plt.text(x_pos, y_pos, parameters, fontsize=30)
    elif x_pos != 0 and y_pos == 0:
        plt.text(x_pos, ytop, parameters, fontsize=30)
    elif x_pos == 0 and y_pos != 0:
        plt.text(xmid, y_pos, parameters, fontsize=30)
    else:
        plt.text(xmid, ytop, parameters, fontsize=30)
    #-----------------------------------------     
    plt.xlabel('Rank of word($x\prime$)', size = 20)
    plt.xscale('log')
    plt.ylabel('$Chain(x\prime)$', size = 20)
    ax.tick_params(axis='x', labelsize=15) 
    ax.tick_params(axis='y', labelsize=15)
    #https://stackoverflow.com/questions/6774086/why-is-my-xlabel-cut-off-in-my-matplotlib-plot
    plt.gcf().subplots_adjust(left = 0.17, bottom = 0.17)
    plt.title(name, fontsize = 20)
    try:
        if Path == '':
            fig.savefig('chain_' + name + '.' + FORMAT, dpi = 500, format = FORMAT)
            plt.show()
        else:
            fig.savefig(Path + 'chain_' + name + '.' + FORMAT, dpi = 500, format = FORMAT)
            plt.close()
    except:
        plt.show()