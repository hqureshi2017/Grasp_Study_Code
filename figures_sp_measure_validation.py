########### HASSAN ##############
#   File contains code for validating Spacing measure.
#   It generates two plots: Difference between perceived and calculated spacing
#   and also the correlation between them
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


def val_diff_sp(df):
        
    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(111)
    
    # PLOTTING difference -24cm
    x_diff = 1
    
    a = df1['val_-24_sp_diff']
    a = [item for sublist in a for item in sublist]
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
    
    a = df1['val_-15_sp_diff']
    a = [item for sublist in a for item in sublist]
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
    x_diff = 3
    
    a = df1['val_24_sp_diff']
    a = [item for sublist in a for item in sublist]
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
    x_diff = 4
    
    a = df1['val_15_sp_diff']
    a = [item for sublist in a for item in sublist]
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
    labels = ['-24','-15','24','15']
    plt.xticks(x, labels, rotation='horizontal')
    plt.title('VALIDATION_SPACING MEASURE DIFF')
    
    
    # Figures 1B
        
    
    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(111)
    
    new = pd.DataFrame()
    new['perceived']= [df1['sp_perceived_c24'].values*-1] + [df1['sp_perceived_c15'].values*-1]  +  [df1['sp_perceived_uc24'].values] + [df1['sp_perceived_uc15'].values]
    new['calculated'] = [df1['sp_calc_c24_total'].values] + [df1['sp_calc_c15_total'].values] + [df1['sp_calc_uc24_total'].values*-1]+ [df1['sp_calc_uc15_total'].values*-1]
    a = new['calculated']
    b = new['perceived']
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
    