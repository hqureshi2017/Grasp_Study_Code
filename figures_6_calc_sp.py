########### HASSAN ##############
#   File contains code for plotting the Calculated vs Perceived Spacing (fig 6)
#   It plots 1) Calculated spacing for Grasp and No Grasp for all locations
#    2) Difference between G and NG for Calculated and Perceived Spacing
#   3) Correlation between Calculated and Perceived Spacing
################################## 

def modules(df):
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

 
    
def calc_sp(df):
    
#    FIGURE 6A
    # NOTE: Calc = Left - Right
    t_val = float(t.ppf([0.975], 30))

    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(111)

    # PLOTTING SPACING -24cm
    x_ng = 0.8
    x_g = 1.2

    dat = []
    a = df['-24_calc_spacing_No_Grasp']
    b = df['-24_calc_spacing_Grasp']
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

    x_val = x_g + max(jit_b) + 0.01
    plt.plot(x_val, mean_b, 'ok', markersize=8)
    plt.plot([x_val, x_val], mean_b_CI, '-k', linewidth=2)
    
    # PLOTTING SPACING -15cm
    x_ng = 1.8
    x_g = 2.2

    dat = []
    a = df['-15_calc_spacing_No_Grasp']
    b = df['-15_calc_spacing_Grasp']
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

    x_val = x_g + max(jit_b) + 0.01
    plt.plot(x_val, mean_b, 'ok', markersize=8)
    plt.plot([x_val, x_val], mean_b_CI, '-k', linewidth=2)

    # PLOTTING SPACING 15cm
    x_ng = 2.8
    x_g = 3.2

    dat = []
    a = [x*-1 for x in df['15_calc_spacing_No_Grasp']]
    b = [x*-1 for x in df['15_calc_spacing_Grasp']]
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

    x_val = x_g + max(jit_b) + 0.01
    plt.plot(x_val, mean_b, 'ok', markersize=8)
    plt.plot([x_val, x_val], mean_b_CI, '-k', linewidth=2)
    
    # PLOTTING SPACING 24cm
    x_ng = 3.8
    x_g = 4.2

    dat = []
    a = [x*-1 for x in df['24_calc_spacing_No_Grasp']]
    b = [x*-1 for x in df['24_calc_spacing_Grasp']]
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

    x_val = x_g + max(jit_b) + 0.01
    plt.plot(x_val, mean_b, 'ok', markersize=8)
    plt.plot([x_val, x_val], mean_b_CI, '-k', linewidth=2)
    
    
    plt.title('CALCULATED SPACING')
    x = [.8 ,1, 1.2,1.8 ,2, 2.2, 2.8 ,3, 3.2,3.8 ,4, 4.2, ]
    labels = ['NG','\n-24','G','NG','\n-15','G', 'NG','\n15','G', 'NG','\n24','G']
    plt.xticks(x, labels, rotation='horizontal')
    ax1.set_xlabel('Horizontal Locations')
    ax1.set_ylabel('Calculated difference in Perceived Spacing (cm)')
    
    
    
def calc_v_perceived_sp(df):
    
    # NEED TO ADD 2nd Y AXIS
    
    # FIGURE 6B
    
    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(111)
    
     # PLOTTING difference Calc Spacing -24
    x_diff = 1
    
    a = df['-24_calc_sp_diff']
#    plt.plot([1]*len(a),a, 'ok', markersize=5, color='r')
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1+x_diff], y1, 'ok', color='r', marker='^')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diff-.05], mean_a, 'ok', markersize=20, marker='^')
    plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)
    
    x_diffr = x_diff+.3
    
    # PLOTTING difference Perceived Spacing -24    
    a = df['-24cm_grasp_vs_nograsp_sp']
#    plt.plot([1]*len(a),a, 'ok', markersize=5, color='r')
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1+x_diffr], y1, 'ok', color='r', marker='^')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diffr-.05], mean_a, 'ok', markersize=20, marker='^')
    plt.plot([x_diffr-.05, x_diffr-.05], mean_a_CI, '-k', linewidth=2)
    
    # PLOTTING difference Calc Spacing -15
    x_diff = x_diff+1
    
    a = df['-15_calc_sp_diff']
#    plt.plot([1]*len(a),a, 'ok', markersize=5, color='r')
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1+x_diff], y1, 'ok', color='r', marker='^')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diff-.05], mean_a, 'ok', markersize=20, marker='^')
    plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)
    
    x_diffr = x_diff+.3
    # PLOTTING difference Perceived Spacing -15    
    
    a = df['-15cm_grasp_vs_nograsp_sp']
#    plt.plot([1]*len(a),a, 'ok', markersize=5, color='r')
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1+x_diffr], y1, 'ok', color='r', marker='^')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diffr-.05], mean_a, 'ok', markersize=20, marker='^')
    plt.plot([x_diffr-.05, x_diffr-.05], mean_a_CI, '-k', linewidth=2)
  
    # PLOTTING difference Calc Spacing 15
    x_diff = x_diff+1
    
    a = df['15_calc_sp_diff']
#    plt.plot([1]*len(a),a, 'ok', markersize=5, color='r')
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1+x_diff], y1, 'ok', color='r', marker='^')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diff-.05], mean_a, 'ok', markersize=20, marker='^')
    plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)
    
    x_diffr = x_diff+.3
    # PLOTTING difference Perceived Spacing 15    
    
    a = df['15cm_grasp_vs_nograsp_sp']
#    plt.plot([1]*len(a),a, 'ok', markersize=5, color='r')
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1+x_diffr], y1, 'ok', color='r', marker='^')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diffr-.05], mean_a, 'ok', markersize=20, marker='^')
    plt.plot([x_diffr-.05, x_diffr-.05], mean_a_CI, '-k', linewidth=2)
  
    # PLOTTING difference Calc Spacing 24
    x_diff = x_diff+1
    
    a = df['24_calc_sp_diff']
#    plt.plot([1]*len(a),a, 'ok', markersize=5, color='r')
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1+x_diff], y1, 'ok', color='r', marker='^')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diff-.05], mean_a, 'ok', markersize=20, marker='^')
    plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)
    
    x_diffr = x_diff+.3
    # PLOTTING difference Perceived Spacing 24    
    
    a = df['24cm_grasp_vs_nograsp_sp']
#    plt.plot([1]*len(a),a, 'ok', markersize=5, color='r')
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1+x_diffr], y1, 'ok', color='r', marker='^')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diffr-.05], mean_a, 'ok', markersize=20, marker='^')
    plt.plot([x_diffr-.05, x_diffr-.05], mean_a_CI, '-k', linewidth=2)
  
    plt.plot([5,.5], [0,0], '--k')    

    plt.title('DIFFERENCE IN SPACING MEASURES')
    x = [.9 ,1.1, 1.3, 1.9 ,2.1, 2.3, 2.9 ,3.1, 3.3, 3.9 ,4.1, 4.3] 
    labels = ['Calc','\n-24','Per','Calc','\n-15','Per', 'Calc','\n15','Per', 'Calc','\n24','Per']
    plt.xticks(x, labels, rotation='horizontal')
    ax1.set_xlabel('Horizontal Locations')
    ax1.set_ylabel('Difference in Perceived Spacing (cm)')
    plt.savefig('./Graphs/data/Fig_6B.pdf')


def calc_vs_perceived_correlation(df):
    
    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(111)
    
    new = pd.DataFrame()
    new['perceived']= [df['grasp_c24cm_sp'].values*-1] + [df['grasp_c15cm_sp'].values*-1]  +  [df['grasp_uc15cm_sp'].values] + [df['grasp_uc24cm_sp'].values]   + [df['nograsp_c24cm_sp'].values*-1] + [df['nograsp_c15cm_sp'].values*-1] + [df['nograsp_uc15cm_sp'].values]  + [df['nograsp_uc24cm_sp'].values]
    new['calculated'] = [df['-24_calc_spacing_Grasp'].values] + [df['-15_calc_spacing_Grasp'].values] + [df['15_calc_spacing_Grasp'].values*-1]+ [df['24_calc_spacing_Grasp'].values*-1] + [df['-24_calc_spacing_No_Grasp'].values] + [df['-15_calc_spacing_No_Grasp'].values] + [df['15_calc_spacing_No_Grasp'].values*-1] + [df['24_calc_spacing_No_Grasp'].values*-1] 
    a = new['perceived']
    b = new['calculated']
    flat_list_a = [item for sublist in a for item in sublist]
    flat_list_b = [item for sublist in b for item in sublist]
        
    plt.scatter(flat_list_b,flat_list_a)    
    
    plt.title('CORRELATION')
    labels =''
    x = []
    plt.xticks(x, labels, rotation='horizontal')
    plt.yticks(x, labels)
    ax1.set_xlabel('Calculation Spacing (cm)')
    ax1.set_ylabel('Perceived Spacing (cm)')    
    
    
    from pydoc import help
    from scipy.stats.stats import pearsonr
    help(pearsonr)
    pearsonr(flat_list_b, flat_list_a)
    
    from scipy.stats import linregress
    linregress(flat_list_b, flat_list_a)
