########### HASSAN ##############
#   File contains code for plotting the Ownership measure (fig 3)
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

def own_outcomes(df):
    # FIGURE 3
    
    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(111)
    
     # PLOTTING -24cm
    x_own = 1
    
    a = df['grasp_c24cm_own']
    jit_a
#    plt.plot([1]*len(a),a, 'ok', markersize=5, color='r')
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1+x_own], y1, 'ok', color='r')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_own-.05], mean_a, 'ok', markersize=20)
    plt.plot([x_own-.05, x_own-.05], mean_a_CI, '-k', linewidth=2)
    
     # PLOTTING -15cm
    x_own = x_own+1
    
    a = df['grasp_c15cm_own']
    jit_a
#    plt.plot([1]*len(a),a, 'ok', markersize=5, color='r')
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1+x_own], y1, 'ok', color='r')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_own-.05], mean_a, 'ok', markersize=20)
    plt.plot([x_own-.05, x_own-.05], mean_a_CI, '-k', linewidth=2)
    
     # PLOTTING 15cm
    x_own = x_own+1
    
    a = df['grasp_uc15cm_own']
    jit_a
#    plt.plot([1]*len(a),a, 'ok', markersize=5, color='r')
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1+x_own], y1, 'ok', color='r')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_own-.05], mean_a, 'ok', markersize=20)
    plt.plot([x_own-.05, x_own-.05], mean_a_CI, '-k', linewidth=2)
    
     # PLOTTING 24cm
    x_own = x_own+1
    
    a = df['grasp_uc24cm_own']
    jit_a
#    plt.plot([1]*len(a),a, 'ok', markersize=5, color='r')
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1+x_own], y1, 'ok', color='r')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_own-.05], mean_a, 'ok', markersize=20)
    plt.plot([x_own-.05, x_own-.05], mean_a_CI, '-k', linewidth=2)
    
    plt.plot([4.2,.8], [4,4], '--k')    

    ax1.set_xlabel('Horizontal Locations')
    ax1.set_ylabel('Perceived Ownership')
    x = [1, 2, 3, 4]
    labels = ['-24','-15','15','24']
    plt.xticks(x, labels, rotation='horizontal')
    y = [0,1,2,3,4,5,6,7,8]
    ylabels = ['','Str Disagree','Disagree','~ Disagree','0','~ Agree', 'Agree', 'Str Agree']
    plt.yticks(y, ylabels, rotation='horizontal')
    plt.title('OWNERSHIP')

def own_diff_outcomes(df):
    # This GRAPH is not actually used. But is a nice way to see the differences
    
    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(111)
    
     # PLOTTING difference -15cm vs -24cm
    x_diff = 1
    
    a = df['-15_vs_-24_own']
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
    
    # PLOTTING difference 15cm vs 24cm
    x_diff = x_diff+1
    
    a = df['15_vs_24_own']
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
    
    # PLOTTING difference -15cm vs 15cm
    x_diff = x_diff+1
    
    a = df['-15_vs_15_own']
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
    
    # PLOTTING difference -24cm vs 24cm
    x_diff = x_diff+1
    
    a = df['-24_vs_24_own']
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
    
    ax1.set_xlabel('Horizontal Locations')
    ax1.set_ylabel('Difference in Perceived Ownership (cm)')
    x = [1, 2, 3, 4]
    labels = ['-15_vs_-24','15_vs_24','-15_vs_15','-24_vs_24']
    plt.xticks(x, labels, rotation='horizontal')
    
    plt.title('OWNERSHIP_diff')