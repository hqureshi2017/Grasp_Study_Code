########### HASSAN ##############
#   File contains code for plotting the Location measure (fig 5)
#   It plots 1) Left hand 2) Right hand and 3) Difference
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
    


#-------------------------
#LOCATION OF LEFT ACROSS G AND NG FOR ALL LOCATIONS
#-------------------------
def loc_outcomes(df):
    
    # FIGURE 5A
    """Function that plots Location of LEFT HAND for Grasp and No Grasp across all 
    trials"""
    t_val = float(t.ppf([0.975], 30))

    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(211)

    # PLOTTING SPACING -24cm
    x_ng = .2
    x_g = .6

    dat = []
    a = list(df['nograsp_c24cm_l'].values-36)
    b = list(df['grasp_c24cm_l'].values-36)
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
    a = list(df['nograsp_c15cm_l'].values-36)
    b = list(df['grasp_c15cm_l'].values-36)
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
    a = [x*-1 for x in list(df['nograsp_uc15cm_l'].values-36)]
    b = [x*-1 for x in list(df['grasp_uc15cm_l'].values-36)]
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
    a = [x*-1 for x in list(df['nograsp_uc24cm_l'].values-36)]
    b = [x*-1 for x in list(df['grasp_uc24cm_l'].values-36)]
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
    labels = ['NG','\n-24','G', 'NG','\n-15','G','NG','\n15','G','NG','\n24','G']
    plt.xticks(x, labels, rotation='horizontal')
    plt.ylim(0,35)

    #-------------------------
    #LOCATION OF RIGHT ACROSS G AND NG FOR ALL LOCATIONS
    #-------------------------
    
    # FIGURE 5B

    """Function that plots Location of RIGHT HAND for Grasp and No Grasp across all 
    trials"""

    t_val = float(t.ppf([0.975], 30))

    ax2 = fig.add_subplot(212)

    # PLOTTING SPACING -24cm
    x_ng = .2
    x_g = .6

    dat = []
    a = list(df['nograsp_c24cm_r'].values-36)
    b = list(df['grasp_c24cm_r'].values-36)
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
    a = list(df['nograsp_c15cm_r'].values-36)
    b = list(df['grasp_c15cm_r'].values-36)
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
    a = list(df['nograsp_uc15cm_r'].values-36)
    b = list(df['grasp_uc15cm_r'].values-36)
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
    a = list(df['nograsp_uc24cm_r'].values-36)
    b = list(df['grasp_uc24cm_r'].values-36)
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
    labels = ['NG','\n-24','G', 'NG','\n-15','G','NG','\n15','G','NG','\n24','G']
    plt.xticks(x, labels, rotation='horizontal')
    ax2.set_xlabel('Horizontal Locations (cm)')
    ax2.set_ylabel('Perceived Location')


    
def location_diff_outcomes(df):
#    ------------
#    # Figure 5C
#    ------------
    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(111)
    
     # PLOTTING difference in location -24cm LEFT
    x_diff = 1
    
    a = df['-24cm_grasp_vs_nograsp_l']
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
    # PLOTTING difference in location -24cm RIGHT
    
    a = df['-24cm_grasp_vs_nograsp_r']
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
    
    # PLOTTING difference in location -15cm LEFT
    x_diff = x_diff+1
    
    a = df['-15cm_grasp_vs_nograsp_l']
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
    # PLOTTING difference in location -15cm RIGHT
    
    a = df['-15cm_grasp_vs_nograsp_r']
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
  
    # PLOTTING difference in location 15cm LEFT
    x_diff = x_diff+1
    
    a = df['15cm_grasp_vs_nograsp_l']
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
    # PLOTTING difference in location 15cm RIGHT
    
    a = df['15cm_grasp_vs_nograsp_r']
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
    
    # PLOTTING difference in location 24cm LEFT
    x_diff = x_diff+1
    
    a = df['24cm_grasp_vs_nograsp_l']
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
    # PLOTTING difference in location 24cm RIGHT
    
    a = df['24cm_grasp_vs_nograsp_r']
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
    
    plt.title('LOCATION_diff')
    x = [.95,1.10,1.3, 1.95,2.10,2.3, 2.95,3.10,3.3, 3.95,4.10,4.3]
    labels = ['L','\n-24','R','L','\n-15','R', 'L','\n15','R', 'L','\n24','R']
    plt.xticks(x, labels, rotation='horizontal')
    ax1.set_xlabel('Horizontal Locations (cm)')
    ax1.set_ylabel('Differences in Perceived Location (cm)')

    