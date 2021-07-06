# -*- coding: utf-8 -*-
"""
@author  gmking

This module is use to calculate the allocations and chains of English corpus.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit

def count_allo(pdframe1, pdframe2, feature1 = "protein", feature2 = "dom", feature3 = "domFreq"):
    '''count the allocations of domains and the chains of proteins
    
    input:
    pdframe1, pdframe2 : protein and dom with class pandas.DataFrame. This two args are from the function, info(file_name, encode = "UTF-8"),  
    in the module count.py.
    
    output:
    add a frame "#allocations" (numbers of allocations of doms) in pdframe2
    add a frame "#chains" (numbers of chains of proteins) in pdframe1
        
    '''
    
    protein_array = pdframe1[feature1] #ex: protein_array=['apple','coffee','elephant']
    dom_array = pdframe2[feature2] #ex: dom_array=['ap', 'ple', 'cof', 'fee', 'e', 'le', 'phant']
    
    #First, we calculate allocations
    
    allocation = {}
    for c in dom_array: 
        allocation[c] = 0
        for w in protein_array:
            t = w.split('-')
            if c in set(t): #ex: 'A' in 'AB', but 'A' not in 'BC'
                allocation[c] += 1

    dom_num_allocations_array = np.array([], dtype = 'int16' )
    
    for i in range(len(pdframe2)):
        dom_num_allocations_array = np.append(dom_num_allocations_array, allocation[dom_array[i]])
    
    #add a frame "#allocations" (numbers of allocations of doms) to dom
    pdframe2['#allocations'] = dom_num_allocations_array 
        
    #Second, we use allocation to calculate chains
    
    
    chain = {}
    for w in protein_array:
        chain[w] = 0
        t = w.split('-')
        for c in set(t):
            #If we don't use set(w) here, the chains will be overcount. 
            #ex: chain('AA') = allocation('A') but not 2*allocation('A')
            chain[w] += allocation[c]
    
    chain_num_array = np.array([], dtype = 'int16')
    
    for i in range(len(pdframe1)):
        chain_num_array = np.append(chain_num_array , chain[protein_array[i]])
    
    #add a frame "#chains" (numbers of chains of proteins) to protein
    pdframe1['#chains'] = chain_num_array 
    

    return None

def Allo_plot(name, dom, x_pos = 0, y_pos = 0, FORMAT = 'png', Path = ''):
    '''draw FRD plot of proteins and domains

       parameters:
    1. name: str
       "XXX" (your file name without filename extension)

    2. dom: pd.daframe
       output of function info() or N_gram_info() in count.py
       you should get them from
       big, dom, protein, longest = info(filename, encode)

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
    Dom = dom.sort_values(by = '#allocations', ascending=False)
    reDom = Dom.reset_index()

    #use OLS to get the fitting parameter
    #-----------------------------------------
    def allo(y, a, b):
        return (a * np.log(y) + b) ** 2

    popt, pcov = curve_fit(allo, dom['domRank'], reDom['#allocations'])
    #popt is the optimal values for the parameters (a,b)
    theo = allo(dom['domRank'], *popt)
    fig, ax = plt.subplots()
    plt.plot(dom['domRank'], theo, 'g--')
    plt.plot(reDom['#allocations'], 'ro', label = 'dom', markersize = 4)
    
    #the following code deal with significant figures of fitting parameters
    #the tutor of significant figures: https://www.usna.edu/ChemDept/_files/documents/manual/apdxB.pdf
    #-----------------------------------------
    allo_dig = len(str(max(reDom['#allocations'])))
    yp_dig = len(str(max(dom['domRank']))) #ln(y') will have yp_dig +1 digits (yp_dig significant figures)
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
    
    a = 1.5  #auto positioning for m = min(dom['domRank']) = 1 always
    b = 2   #auto positioning for M = max(dom['domRank'])
    xmid = max(dom['domRank'])**(b/(a+b))  #exp([a*log(m)+b*log(M)]/[a+b]) = m^(a/[a+b]) * M^(b/[a+b])    
    ytop = max(reDom['#allocations'])*5/7
    
    if x_pos != 0 and y_pos != 0:
        plt.text(x_pos, y_pos, parameters, fontsize=30)
    elif x_pos != 0 and y_pos == 0:
        plt.text(x_pos, ytop, parameters, fontsize=30)
    elif x_pos == 0 and y_pos != 0:
        plt.text(xmid, y_pos, parameters, fontsize=30)
    else:
        plt.text(xmid, ytop, parameters, fontsize=30)
    #-----------------------------------------
    plt.xlabel('rank of domain($y\prime$)', size = 20)
    plt.xscale('log')
    plt.ylabel('allocations', size = 20)
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
        
def Chain_plot(name, protein, x_pos = 0, y_pos = 0, FORMAT = 'png', Path = ''):
    '''draw allocation-rank plot 

       parameters:
    1. name: str
       "XXX" (your file name without filename extension)

    2. protein: pd.daframe
       output of function info() or N_gram_info() in count.py
       you should get them from
       big, dom, protein, longest = info(filename, encode)

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
    Protein = protein.sort_values(by='#chains', ascending=False)
    reProtein = Protein.reset_index()

    #use OLS to get the fitting parameter
    #-----------------------------------------
    def chain(x, a, b):
        return (a * np.log(x) + b)

    popt, pcov = curve_fit(chain, protein['proteinRank'], reProtein['#chains'])
    #popt is the optimal values for the parameters (a,b)
    theo = chain(protein['proteinRank'], *popt)
    fig, ax = plt.subplots()
    plt.plot(protein['proteinRank'], theo, 'g--')
    plt.plot(reProtein['#chains'], 'ro', label = 'protein', markersize = 4)


    
    #the following code deal with significant figures of fitting parameters
    #the tutor of significant figures: https://www.usna.edu/ChemDept/_files/documents/manual/apdxB.pdf
    #-----------------------------------------
    chain_dig = len(str(max(reProtein['#chains'])))
    xp_dig = len(str(max(protein['proteinRank']))) #ln(x') will have xp_dig +1 digits (xp_dig significant figures)
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
    
    a = 1.5  #auto positioning for m = min(dom['domRank']) = 1 always
    b = 2   #auto positioning for M = max(dom['domRank'])
    xmid = max(protein['proteinRank']) ** (b/(a+b))  #exp([a*log(m)+b*log(M)]/[a+b]) = m^(a/[a+b]) * M^(b/[a+b])
    ytop = max(reProtein['#chains'])*5/7
    
    if x_pos != 0 and y_pos != 0:
        plt.text(x_pos, y_pos, parameters, fontsize=30)
    elif x_pos != 0 and y_pos == 0:
        plt.text(x_pos, ytop, parameters, fontsize=30)
    elif x_pos == 0 and y_pos != 0:
        plt.text(xmid, y_pos, parameters, fontsize=30)
    else:
        plt.text(xmid, ytop, parameters, fontsize=30)
    #-----------------------------------------     
    plt.xlabel('rank of protein($x\prime$)', size = 20)
    plt.xscale('log')
    plt.ylabel('chains', size = 20)
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