########### HASSAN ##############
#   File contains code for plotting the Spacing measure (fig 2)
#   It contains two functions: 1) Plots the Perceived Spacing for Grasp and No
#   Grasp across all 4 locations.
#   2) Plots the difference between the Grasp and No Grasp for each location
#   
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


def sp_outcomes(df):
    #    Function that plots the difference between Grasp and No Grasp across all 
    #    trials
    
    t_val = float(t.ppf([0.975], 30))

    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(111)

    # PLOTTING SPACING -24cm
    x_ng = 0.8
    x_g = 1.2

    dat = []
    a = list(df['nograsp_c24cm_sp'].values * -1)
    b = list(df['grasp_c24cm_sp'].values * -1)
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

    x_val = x_g + max(jit_b) + 0.1
    plt.plot(x_val, mean_b, 'ok', markersize=8, color='r')
    plt.plot([x_val, x_val], mean_b_CI, '-k', linewidth=2)

    # PLOTTING SPACING -15cm
    x_ng = 1.8
    x_g = 2.2

    dat = []
    a = list(df['nograsp_c15cm_sp'].values * -1)
    b = list(df['grasp_c15cm_sp'].values * -1)
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

    x_val = x_g + max(jit_b) + 0.1
    plt.plot(x_val, mean_b, 'ok', markersize=8,color='r')
    plt.plot([x_val, x_val], mean_b_CI, '-k', linewidth=2)

    # PLOTTING SPACING 15cm
    x_ng = 2.8
    x_g = 3.2

    dat = []
    a = list(df['nograsp_uc15cm_sp'].values)
    b = list(df['grasp_uc15cm_sp'].values)
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

    x_val = x_g + max(jit_b) + 0.1
    plt.plot(x_val, mean_b, 'ok', markersize=8, color='r')
    plt.plot([x_val, x_val], mean_b_CI, '-k', linewidth=2)

    # PLOTTING SPACING 24cm
    x_ng = 3.8
    x_g = 4.2

    dat = []
    a = list(df['nograsp_uc24cm_sp'].values)
    b = list(df['grasp_uc24cm_sp'].values)
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

    x_val = x_g + max(jit_b) + 0.1
    plt.plot(x_val, mean_b, 'ok', markersize=8, color='r')
    plt.plot([x_val, x_val], mean_b_CI, '-k', linewidth=2)

    ax1.set_xlabel('Horizontal Locations')
    ax1.set_ylabel('Perceived Spacing (cm)')
    
    x = [.8, 1.05, 1.3, 1.8, 2.05, 2.3, 2.8, 3.05, 3.3, 3.8, 4.05, 4.3]
    labels = ['NG','\n-24','G', 'NG','\n-15','G','NG','\n15','G','NG','\n24','G']
    plt.xticks(x, labels, rotation='horizontal')
    plt.title('PERCEIVED SPACING')


    # FIGURE 2B - Difference
       
def sp_diff_outcomes(df):
    
    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(111)
    
    # PLOTTING difference -24cm
    x_diff = 1
    
    a = df['-24cm_grasp_vs_nograsp_sp']
#    plt.plot([1]*len(a),a, 'ok', markersize=5, color='r')
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1+x_diff], y1, 'ok', marker='^')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diff-.05], mean_a, 'ok', markersize=20, marker='^')
    plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)
    
    
     # PLOTTING difference -15cm
    x_diff = 2

    a = df['-15cm_grasp_vs_nograsp_sp']
#    plt.plot([1]*len(a),a, 'ok', markersize=5, color='r')
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1+x_diff], y1, 'ok', marker='^')
    
    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diff-.05], mean_a, 'ok', markersize=20, marker='^')
    plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)
    
    
    # PLOTTING difference 15cm
    x_diff = 3
    
    a = df['15cm_grasp_vs_nograsp_sp']
#    plt.plot([1]*len(a),a, 'ok', markersize=5, color='r')
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1+x_diff], y1, 'ok', marker='^')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diff-.05], mean_a, 'ok', markersize=20, marker='^')
    plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)
    
    # PLOTTING difference 24cm
    x_diff = 4
    
    a = df['24cm_grasp_vs_nograsp_sp']
#    plt.plot([1]*len(a),a, 'ok', markersize=5, color='r')
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1+x_diff], y1, 'ok', marker='^')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diff-.05], mean_a, 'ok', markersize=20, marker='^')
    plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)

    plt.plot([4.2,.8], [0,0], '--k')    

    ax1.set_xlabel('Horizontal Locations')
    ax1.set_ylabel('Difference in Perceived Spacing (cm)')
    x = [1, 2, 3, 4]
    labels = ['-24','-15','15','24']
    plt.xticks(x, labels, rotation='horizontal')
    plt.title('SPACING_diff')
    