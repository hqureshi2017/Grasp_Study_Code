########### HASSAN ##############
#   File contains code for validating Spacing measure.
#   It generates two plots: Difference between perceived and calculated spacing
#   and also the correlation between them


#   See below also for valdiation on NG condition   
################################## 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import seaborn as sns
import random
from scipy.stats import t
from math import sqrt
from numpy import max
from numpy import mean
from numpy import std
from jitter import jitter
from grasp_study_code import text_figure

def val_diff_sp(df):

    # FIGURE 1A
    
    # Difference between calc and per spacing in the validation
        
    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(111)
    
    t_val = float(t.ppf([0.975], 30))

        
    text_figure('Fig1A',['val_-24_sp_diff_1', 'val_-24_sp_diff_2',
                         'val_-15_sp_diff_1', 'val_-15_sp_diff_2',
                         'val_15_sp_diff_1', 'val_15_sp_diff_2',
                         'val_24_sp_diff_1', 'val_24_sp_diff_2'], df)
    
    # PLOTTING difference -24cm
    x_diff = 1
    
    a = [df['val_-24_sp_diff_1'].values] + [df['val_-24_sp_diff_2'].values] 
    b = [df['val_-24_sp_diff_1'].values] + [df['val_-24_sp_diff_2'].values] 
    a = [item for sublist in a for item in sublist]
    b = [item for sublist in b for item in sublist]
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = jitter(dat)
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1+x_diff], y1, 'ok', marker='^')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diff-.05], mean_a, 'ok', markersize=10, marker='^')
    plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)
    
        
    # PLOTTING difference -15cm
    x_diff = 2
    
    a = [df['val_-15_sp_diff_1'].values] + [df['val_-15_sp_diff_2'].values] 
    b = [df['val_-15_sp_diff_1'].values] + [df['val_-15_sp_diff_2'].values] 
    a = [item for sublist in a for item in sublist]
    b = [item for sublist in b for item in sublist]
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = jitter(dat)
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1+x_diff], y1, 'ok', marker='^')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diff-.05], mean_a, 'ok', markersize=10, marker='^')
    plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)
    
    
    # PLOTTING difference 15cm
    x_diff = 3
    
    a = [df['val_15_sp_diff_1'].values] + [df['val_15_sp_diff_2'].values] 
    b = [df['val_15_sp_diff_1'].values] + [df['val_15_sp_diff_2'].values] 
    a = [item for sublist in a for item in sublist]
    b = [item for sublist in b for item in sublist]
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = jitter(dat)
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1+x_diff], y1, 'ok', marker='^')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diff-.05], mean_a, 'ok', markersize=10, marker='^')
    plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)
    
    # PLOTTING difference 24cm
    x_diff = 4
    
    a = [df['val_24_sp_diff_1'].values] + [df['val_24_sp_diff_2'].values] 
    b = [df['val_24_sp_diff_1'].values] + [df['val_24_sp_diff_2'].values] 
    a = [item for sublist in a for item in sublist]
    b = [item for sublist in b for item in sublist]
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = jitter(dat)
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1+x_diff], y1, 'ok', marker='^')
        
    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diff-.05], mean_a, 'ok', markersize=10, marker='^')
    plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)
    
    
    plt.plot([4.2,.8], [0,0], '--k')    

    # Hide the right and top spines
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
        
    #Removing ticks from axes
    ax1.tick_params( axis='x', which='major', bottom='off', top='off', labelbottom='on', left='off', right='off', labelsize=14)
    ax1.tick_params( axis='y', which='major', direction='out', length=3, bottom='off', top='off', labelbottom='on', right='off', labelsize=14)
    
    xticks = [1, 1.5, 2, 3, 3.5, 4 ]
    labels = ['-24 cm', 'Crossed', '-15 cm',
              '15 cm', 'Uncrossed','24 cm']
    plt.xticks(xticks, labels)
    
    # vertical alignment of xtick labels
    va = [ 0, -.05, 0, 0, -.05, 0, 
          0, -.05, 0, 0, -.05, 0, ]
    for z, y in zip( ax1.get_xticklabels( ), va ):
        z.set_y( y )
        
    ax1.set_xlabel('')
    ax1.set_ylabel('Difference in spacing (cm)',size=20)
    
    plt.title('')
    plt.savefig('./Graphs/data/' + 'fig_2C.pdf')
    plt.savefig('./Graphs/data/' + 'fig_2C.svg')

    
  
    
def val_NG(df):
    
    # FIGURE 
    
    # Difference between calc and per spacing in the NO GRASP condition
        
    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(111)
    
    t_val = float(t.ppf([0.975], 30))

        
        
    # PLOTTING difference -24cm
    x_diff = 1
    
    a = [df['-24_calc_spacing_No_Grasp']-df['nograsp_c24cm_sp']]
    b = [df['-24_calc_spacing_No_Grasp']-df['nograsp_c24cm_sp']]
    a = [item for sublist in a for item in sublist]
    b = [item for sublist in b for item in sublist]
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = jitter(dat)
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1+x_diff], y1, 'ok', marker='^')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diff-.05], mean_a, 'ok', markersize=10, marker='^')
    plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)
    
        
    # PLOTTING difference -15cm
    x_diff = 2
    
    a = [df['-15_calc_spacing_No_Grasp']-df['nograsp_c15cm_sp']]
    b = [df['-15_calc_spacing_No_Grasp']-df['nograsp_c15cm_sp']]
    a = [item for sublist in a for item in sublist]
    b = [item for sublist in b for item in sublist]
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = jitter(dat)
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1+x_diff], y1, 'ok', marker='^')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diff-.05], mean_a, 'ok', markersize=10, marker='^')
    plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)
    
    
    # PLOTTING difference 15cm
    x_diff = 3
    
    a = [df['15_calc_spacing_No_Grasp']-df['nograsp_uc15cm_sp']]
    b = [df['15_calc_spacing_No_Grasp']-df['nograsp_uc15cm_sp']]
    a = [item for sublist in a for item in sublist]
    b = [item for sublist in b for item in sublist]
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = jitter(dat)
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1+x_diff], y1, 'ok', marker='^')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diff-.05], mean_a, 'ok', markersize=10, marker='^')
    plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)
    
    # PLOTTING difference 24cm
    x_diff = 4
    
    a = [df['24_calc_spacing_No_Grasp']-df['nograsp_uc24cm_sp']]
    b = [df['24_calc_spacing_No_Grasp']-df['nograsp_uc24cm_sp']]
    a = [item for sublist in a for item in sublist]
    b = [item for sublist in b for item in sublist]
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = jitter(dat)
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1+x_diff], y1, 'ok', marker='^')
        
    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diff-.05], mean_a, 'ok', markersize=10, marker='^')
    plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)
    
    
    plt.plot([4.2,.8], [0,0], '--k')    

    # Hide the right and top spines
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
        
    #Removing ticks from axes
    ax1.tick_params( axis='x', which='minor', direction='out', length=60, top='off', right='off', labelbottom = 'on' )
    ax1.tick_params( axis='x', which='major', bottom='off', top='off', labelbottom='on', left='off', right='off')
    ax1.tick_params( axis='y', which='both', direction='out', length=3, bottom='off', top='off', labelbottom='on', left='on', right='off')

    xticks = [1, 1.5, 2, 3, 3.5, 4 ]
    labels = ['-24 cm', 'crossed', '-15 cm',
              '15 cm', 'uncrossed','24 cm']
    plt.xticks(xticks, labels)
    
    #Removing ticks from axes
    ax1.tick_params( axis='x', which='major', bottom='off', top='off', labelbottom='on', left='off', right='off', labelsize=14)
    ax1.tick_params( axis='y', which='major', direction='out', length=3, bottom='off', top='off', labelbottom='on', right='off', labelsize=14)
    
    # vertical alignment of xtick labels
    va = [ 0, -.05, 0, 0, -.05, 0, 
          0, -.05, 0, 0, -.05, 0, ]
    for z, y in zip( ax1.get_xticklabels( ), va ):
        z.set_y( y )
        
    ax1.set_xlabel('')
    ax1.set_ylabel('Difference in spacing (cm)', size=20)
    
    plt.title('')
    plt.savefig('./Graphs/data/' + 'diff_sp_NG.pdf')
    plt.savefig('./Graphs/data/' + 'diff_sp_NG.svg')

   
    
    
    
    
        
def val_diff_sp_correlation(df):
   
    # Figures 1B
    
    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(111)
    t_val = float(t.ppf([0.975], 30))
    
    calc = [[df['sp_calc_c24_1'].values] + [df['sp_calc_c24_2'].values] 
            + [df['sp_calc_c15_1'].values] + [df['sp_calc_c15_2'].values] 
            + [df['sp_calc_uc24_1'].values] + [df['sp_calc_uc24_2'].values] 
            + [df['sp_calc_uc15_1'].values] + [df['sp_calc_uc15_2'].values]]
   
    per = [[df['R24_1'].values] + [df['R24_2'].values] +
            [df['R15_1'].values] + [df['R15_2'].values] +
            [df['L24_1'].values] + [df['L24_2'].values] +
            [df['L15_1'].values] + [df['L15_2'].values] ]
            
    a = [item for sublist in calc for item in sublist]
    b = [item for sublist in per for item in sublist]
    a = [item for sublist in a for item in sublist]
    b = [item for sublist in b for item in sublist]
   
    #r, p = scipy.stats.pearsonr(a, b)

    fit = np.polyfit(b,a,1)
    fit_fn = np.poly1d(fit)
    plt.scatter(b,a)
#    sns.swarmplot(b,a, color='k')    
    plt.plot(a, fit_fn(a), linestyle='-',
             color='k',linewidth=1)
    
    plt.title('CORRELATION between Perceived and Calculated Spacing_VALIDATION\n r = 0.64  p = 2.23')
    labels =''
    ax1.grid( 'off', axis='both' )
    ax1.set_xlabel('Calculated Spacing (cm)')
    ax1.set_ylabel('Perceived Spacing (cm)')    
    plt.savefig( './Graphs/data/' + 'fig_1B.pdf')
    plt.savefig( './Graphs/data/' + 'fig_1B.svg')

    


    from pydoc import help
    from scipy.stats.stats import pearsonr
    #help(pearsonr)
    pearsonr(b,a)
    
    from scipy.stats import linregress
    linregress(b, a)
    
    
    
    
    
    
    
    
    
def random(df):    
    # BELOW GRAPH - NOT USED. Just want to see what the location measure looks
#    like for validation


    # FIGURE --
    """Function that plots Location of LEFT HAND for Grasp and No Grasp across all 
    trials"""
    t_val = float(t.ppf([0.975], 30))

    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(211)

    # PLOTTING SPACING -24cm
    x_ng = .2
    x_g = .6

    dat = []
    a = list(df['loc_L_R24_1'])
    b = list(df['loc_L_R24_2'])
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = _jitter(dat)
    for x1, x2, y1, y2 in zip(jit_a, jit_b, a, b):
        plt.plot([x1 + x_ng, x2 + x_g], [y1, y2], '-', color='lightgrey')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot(x_ng, mean_a, 'ok', markersize=8)
    plt.plot([x_ng, x_ng], mean_a_CI, '-k', linewidth=2)

    mean_b = mean(b)
    sd_b = std(b)
    n = 30
    mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
                 mean_b + (sd_b / sqrt(n)) * t_val]
    x_val = x_g + max(jit_b) + .01
    plt.plot(x_val, mean_b, 'ok', markersize=8)
    plt.plot([x_val, x_val], mean_b_CI, '-k', linewidth=2)

    # PLOTTING SPACING -15cm
    x_ng = x_ng+1
    x_g = x_g+1

    dat = []
    a = list(df['loc_L_R15_1'])
    b = list(df['loc_L_R15_2'])
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = _jitter(dat)
    for x1, x2, y1, y2 in zip(jit_a, jit_b, a, b):
        plt.plot([x1 + x_ng, x2 + x_g], [y1, y2], '-', color='lightgrey')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot(x_ng, mean_a, 'ok', markersize=8)
    plt.plot([x_ng, x_ng], mean_a_CI, '-k', linewidth=2)

    mean_b = mean(b)
    sd_b = std(b)
    n = 30
    mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
                 mean_b + (sd_b / sqrt(n)) * t_val]
    x_val = x_g + max(jit_b) + .01
    plt.plot(x_val, mean_b, 'ok', markersize=8)
    plt.plot([x_val, x_val], mean_b_CI, '-k', linewidth=2)

    # PLOTTING SPACING 15cm
    x_ng = x_ng+1
    x_g = x_g+1

    dat = []
    a = [x*-1 for x in list(df['loc_L_L15_1'])]
    b = [x*-1 for x in list(df['loc_L_L15_2'])]
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = _jitter(dat)
    for x1, x2, y1, y2 in zip(jit_a, jit_b, a, b):
        plt.plot([x1 + x_ng, x2 + x_g], [y1, y2], '-', color='lightgrey')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot(x_ng, mean_a, 'ok', markersize=8)
    plt.plot([x_ng, x_ng], mean_a_CI, '-k', linewidth=2)

    mean_b = mean(b)
    sd_b = std(b)
    n = 30
    mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
                 mean_b + (sd_b / sqrt(n)) * t_val]
    x_val = x_g + max(jit_b) + .01
    plt.plot(x_val, mean_b, 'ok', markersize=8)
    plt.plot([x_val, x_val], mean_b_CI, '-k', linewidth=2)

    # PLOTTING SPACING 24cm
    x_ng = x_ng+1
    x_g = x_g+1

    dat = []
    a = [x*-1 for x in list(df['loc_L_L24_1'])]
    b = [x*-1 for x in list(df['loc_L_L24_2'])]
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = _jitter(dat)
    for x1, x2, y1, y2 in zip(jit_a, jit_b, a, b):
        plt.plot([x1 + x_ng, x2 + x_g], [y1, y2], '-', color='lightgrey')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot(x_ng, mean_a, 'ok', markersize=8)
    plt.plot([x_ng, x_ng], mean_a_CI, '-k', linewidth=2)

    mean_b = mean(b)
    sd_b = std(b)
    n = 30
    mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
                 mean_b + (sd_b / sqrt(n)) * t_val]
    x_val = x_g + max(jit_b) + .01
    plt.plot(x_val, mean_b, 'ok', markersize=8)
    plt.plot([x_val, x_val], mean_b_CI, '-k', linewidth=2)
    
    ax1.set_xlabel('')
    ax1.set_ylabel('Perceived Location (cm)')
    plt.title('LEFT')
    x = [.2,.4,.6, 1.2,1.4,1.6, 2.2,2.4,2.6, 3.2,3.4,3.6]
    labels = ['1','\n-24','2', '1','\n-15','2','1','\n15','2','1','\n24','2']
    plt.xticks(x, labels, rotation='horizontal')
    plt.ylim(0,35)
    
    
    
    #-------------------------
    #LOCATION OF RIGHT ACROSS G AND NG FOR ALL LOCATIONS
    #-------------------------
    
    # FIGURE --

    """Function that plots Location of RIGHT HAND for Grasp and No Grasp across all 
    trials"""

    t_val = float(t.ppf([0.975], 30))

    ax2 = fig.add_subplot(212)

    # PLOTTING SPACING -24cm
    x_ng = .2
    x_g = .6

    dat = []
    a = list(df['loc_R_R24_1'])
    b = list(df['loc_R_R24_2'])
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = _jitter(dat)
    for x1, x2, y1, y2 in zip(jit_a, jit_b, a, b):
        plt.plot([x1 + x_ng, x2 + x_g], [y1, y2], '-', color='lightgrey')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot(x_ng, mean_a, 'ok', markersize=8)
    plt.plot([x_ng, x_ng], mean_a_CI, '-k', linewidth=2)

    mean_b = mean(b)
    sd_b = std(b)
    n = 30
    mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
                 mean_b + (sd_b / sqrt(n)) * t_val]
    x_val = x_g + max(jit_b) + .01
    plt.plot(x_val, mean_b, 'ok', markersize=8)
    plt.plot([x_val, x_val], mean_b_CI, '-k', linewidth=2)

    # PLOTTING SPACING -15cm
    x_ng = x_ng+1
    x_g = x_g+1

    dat = []
    a = list(df['loc_R_R15_1'])
    b = list(df['loc_R_R15_2'])
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = _jitter(dat)
    for x1, x2, y1, y2 in zip(jit_a, jit_b, a, b):
        plt.plot([x1 + x_ng, x2 + x_g], [y1, y2], '-', color='lightgrey')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot(x_ng, mean_a, 'ok', markersize=8)
    plt.plot([x_ng, x_ng], mean_a_CI, '-k', linewidth=2)

    mean_b = mean(b)
    sd_b = std(b)
    n = 30
    mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
                 mean_b + (sd_b / sqrt(n)) * t_val]
    x_val = x_g + max(jit_b) + .01
    plt.plot(x_val, mean_b, 'ok', markersize=8)
    plt.plot([x_val, x_val], mean_b_CI, '-k', linewidth=2)

    # PLOTTING SPACING 15cm
    x_ng = x_ng+1
    x_g = x_g+1

    dat = []
    a = list(df['loc_R_L15_1'])
    b = list(df['loc_R_L15_2'])
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = _jitter(dat)
    for x1, x2, y1, y2 in zip(jit_a, jit_b, a, b):
        plt.plot([x1 + x_ng, x2 + x_g], [y1, y2], '-', color='lightgrey')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot(x_ng, mean_a, 'ok', markersize=8)
    plt.plot([x_ng, x_ng], mean_a_CI, '-k', linewidth=2)

    mean_b = mean(b)
    sd_b = std(b)
    n = 30
    mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
                 mean_b + (sd_b / sqrt(n)) * t_val]
    x_val = x_g + max(jit_b) + .01
    plt.plot(x_val, mean_b, 'ok', markersize=8)
    plt.plot([x_val, x_val], mean_b_CI, '-k', linewidth=2)

    # PLOTTING SPACING 24cm
    x_ng = x_ng+1
    x_g = x_g+1

    dat = []
    a = list(df['loc_R_L24_1'])
    b = list(df['loc_R_L24_2'])
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = _jitter(dat)
    for x1, x2, y1, y2 in zip(jit_a, jit_b, a, b):
        plt.plot([x1 + x_ng, x2 + x_g], [y1, y2], '-', color='lightgrey')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot(x_ng, mean_a, 'ok', markersize=8)
    plt.plot([x_ng, x_ng], mean_a_CI, '-k', linewidth=2)

    mean_b = mean(b)
    sd_b = std(b)
    n = 30
    mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
                 mean_b + (sd_b / sqrt(n)) * t_val]
    x_val = x_g + max(jit_b) + .01
    plt.plot(x_val, mean_b, 'ok', markersize=8)
    plt.plot([x_val, x_val], mean_b_CI, '-k', linewidth=2)
        
    plt.title('RIGHT')
    x = [.2,.4,.6, 1.2,1.4,1.6, 2.2,2.4,2.6, 3.2,3.4,3.6]
    labels = ['1','\n-24','2', '1','\n-15','2','1','\n15','2','1','\n24','2']
    plt.xticks(x, labels, rotation='horizontal')
    plt.ylim(-10,15)
