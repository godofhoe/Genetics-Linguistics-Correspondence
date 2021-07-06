import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_EXCEL(filename, EXCEL_load):
    if 'xlsx' in filename.split('.'):
        df = pd.read_excel(EXCEL_load)
        print('read %s successfully' % filename)
    if 'csv' in filename.split('.'):
        df = pd.read_csv(EXCEL_load)
        print('read %s successfully' % filename)
    return df

def AGLMV(name, SP, FORMAT, df, filename = 'AGLMV.xlsx', Path = ''):
    '''plot r_f of your data
    
    ------paras
    name: str
       name of your AGMV plot
    SP: str, should be 'T' or others
       If you don't want to save picture, just assign SP != 'T'.
    FORMAT: str
       format of your picture
    df: panda.DataFrame
       output of read_EXCEL
    filename: str
       filename of your data. please read 'readme.png' to know the format of your file  
       ######support file format: csv, excel
       
    ------output
    a picture
    '''
    
    if (SP == 'T') and (FORMAT == 'png' or 'pdf' or 'ps' or 'eps' or 'svg'):  
        maker_list = ['D', 'o', 'v', '^', '<', '>', '1', '2', '3', '4']
        algo = [i for i in df['name']] #name of algorithm
        GLMV = [i for i in df['GL/MV']] #G/MV_1 of algorithm
        A = [i for i in df['Accuracy']] #Accuracy of algorithm

        fig, ax = plt.subplots()
        for i in range(len(algo)):
            x = GLMV[i]
            y = A[i]
            plt.plot(x, y, maker_list[i], label = algo[i], markersize = 8)  

        ax.tick_params(axis='x', labelsize=15) 
        ax.tick_params(axis='y', labelsize=15)
        #https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.ticklabel_format.html
        ax.ticklabel_format(axis='x', style='sci',scilimits=(0,2))
        plt.ylim([0, 1.1])  
        x_m = min(GLMV)
        x_M = max(GLMV)
        plt.xlim([0.95*x_m, 1.05*x_M])
        plt.title(name, size = 20)
        plt.xlabel('$GL^{0.15}/MV_1$', size = 20)
        plt.ylabel('Accuracy', size = 20)
        #https://stackoverflow.com/questions/6774086/why-is-my-xlabel-cut-off-in-my-matplotlib-plot
        plt.gcf().subplots_adjust(left = 0.17, bottom = 0.17)
        plt.legend(loc = 'best', prop = {'size': 15})

        fig.savefig(Path + 'AGLMV of ' + name + '.' + FORMAT, dpi = 300, format = FORMAT)
        plt.close()
        print('save figure successfully !')
    else:
        print('FORMAT is wrong or NO AGMV should be saved.')