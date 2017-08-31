########### HASSAN ##############
#   File contains code for Comparisons of Spacing and Ownership between specific 
#    locations.
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

         
#    ----------------------------------
#    NEED TO PLOT FIGURES 4  - secondary y axis
#    ---------------------------------
    
    # ----------------
    # 4.1 + 4.2
    # ----------------

def distance_outcomes(df):
#   FIGURE 4.1A - difference in perceived sp -15 vs -24

    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(221)
    
     # PLOTTING -15
    x_diff = 1
    a = df['-15cm_grasp_vs_nograsp_sp']
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
    
    x_diffr = x_diff+.5
    
    # PLOTTING -24    
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
    
    plt.plot([.8,1.7], [0,0], '--k')    

    # Plotting difference
    x_diffr = x_diff+1
    a1 = df['-15_vs_-24_sp']
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1+x_diffr], y1, 'ok', color='r', marker='^')
    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diffr-.05], mean_a, 'ok', markersize=20, marker='^')
    plt.plot([x_diffr-.05, x_diffr-.05], mean_a_CI, '-k', linewidth=2)
        
        
    plt.title('DIFFERENCE IN PERCEIVED SPACING')
    x = [1, 1.5]
    labels = '-15','-24', 'diff'
    plt.xticks(x, labels, rotation='horizontal')
    ax1.set_xlabel('')
    ax1.set_ylabel('Difference in Perceived Spacing (cm)')    
    plt.ylim(-20,25)

#   FIGURE 4.2A - difference in perceived sp 15 vs 24

    ax2 = fig.add_subplot(222)
    
     # PLOTTING 15
    x_diff = 1
    a = df['15cm_grasp_vs_nograsp_sp']
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
    
    x_diffr = x_diff+.5
    
    # PLOTTING 24    
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
    
    plt.plot([.8,1.7], [0,0], '--k')    

    # Plotting difference
#    x_diffr = x_diff+1
#    a = df['15_vs_24_sp']
#    for x1, y1, in zip(jit_a, a):
#        plt.plot([x1+x_diffr], y1, 'ok', color='r', marker='^')
#    mean_a = mean(a)
#    sd_a = std(a)
#    n = 30
#    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
#                 mean_a + (sd_a / sqrt(n)) * t_val]
#    plt.plot([x_diffr-.05], mean_a, 'ok', markersize=20, marker='^')
#    plt.plot([x_diffr-.05, x_diffr-.05], mean_a_CI, '-k', linewidth=2)
    
    
    plt.title('DIFFERENCE IN PERCEIVED SPACING')
    x = [1, 1.5]
    labels = '15','24', 'diff'
    plt.xticks(x, labels, rotation='horizontal')
    ax2.set_xlabel('')
    ax2.set_ylabel('Difference in Perceived Spacing (cm)')
    plt.ylim(-20,25)


#   FIGURE 4.1B - difference in perceived own -15 vs -24

    ax3 = fig.add_subplot(223)
    
     # PLOTTING -15
    x_diff = 1
    a = df['grasp_c15cm_own']
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
    
    x_diffr = x_diff+.5
    
    # PLOTTING -24    
    a = df['grasp_c24cm_own']
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
    
    plt.plot([.8,1.7], [4,4], '--k')    

     # Plotting difference
#    x_diffr = x_diff+1
#    a = df['-15_vs_-24_own']
#    for x1, y1, in zip(jit_a, a):
#        plt.plot([x1+x_diffr], y1, 'ok', color='r', marker='^')
#    mean_a = mean(a)
#    sd_a = std(a)
#    n = 30
#    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
#                 mean_a + (sd_a / sqrt(n)) * t_val]
#    plt.plot([x_diffr-.05], mean_a, 'ok', markersize=20, marker='^')
#    plt.plot([x_diffr-.05, x_diffr-.05], mean_a_CI, '-k', linewidth=2)
    
    
    plt.title('DIFFERENCE IN PERCEIVED OWNERSHIP')
    x = [1, 1.5]
    labels = '-15','-24', 'diff'
    plt.xticks(x, labels, rotation='horizontal')
    y = [0,1,2,3,4,5,6,7,8]
    ylabels = ['','Str Disagree','Disagree','~ Disagree','0','~ Agree', 'Agree', 'Str Agree']
    plt.yticks(y, ylabels, rotation='horizontal')
    ax3.set_xlabel('Horizontal Locations')
    ax3.set_ylabel('Difference in Perceived Spacing (cm)')


#   FIGURE 4.2B - difference in perceived own 15 vs 24

    ax4 = fig.add_subplot(224)
    
     # PLOTTING 15
    x_diff = 1
    a = df['grasp_uc15cm_own']
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
    
    x_diffr = x_diff+.5
    
    # PLOTTING 24    
    a = df['grasp_uc24cm_own']
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
    
    plt.plot([.8,1.7], [4,4], '--k')    

    # Plotting difference
#    x_diffr = x_diff+1
#    a = df['15_vs_24_own']
#    for x1, y1, in zip(jit_a, a):
#        plt.plot([x1+x_diffr], y1, 'ok', color='r', marker='^')
#    mean_a = mean(a)
#    sd_a = std(a)
#    n = 30
#    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
#                 mean_a + (sd_a / sqrt(n)) * t_val]
#    plt.plot([x_diffr-.05], mean_a, 'ok', markersize=20, marker='^')
#    plt.plot([x_diffr-.05, x_diffr-.05], mean_a_CI, '-k', linewidth=2)
    
    
    plt.title('DIFFERENCE IN PERCEIVED OWNERSHIP')
    x = [1, 1.5]
    labels = '15','24', 'diff'
    plt.xticks(x, labels, rotation='horizontal')
    y = [0,1,2,3,4,5,6,7,8]
    ylabels = ['','Str Disagree','Disagree','~ Disagree','0','~ Agree', 'Agree', 'Str Agree']
    plt.yticks(y, ylabels, rotation='horizontal')
    ax4.set_xlabel('Horizontal Locations')
    ax4.set_ylabel('')
    

    # ----------------
    # 4.3 + 4.4
    # ----------------

def midline_outcomes(df):
#   FIGURE 4.3A - difference in perceived sp -15 vs 15

    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(221)
    
     # PLOTTING -15
    x_diff = 1
    a = df['-15cm_grasp_vs_nograsp_sp']
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
    
    x_diffr = x_diff+.5
    
    # PLOTTING 15    
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
    
    plt.plot([.8,1.7], [0,0], '--k')    

    # Plotting difference
#    x_diffr = x_diff+1
#    a = df['-15_vs_15_sp']
#    for x1, y1, in zip(jit_a, a):
#        plt.plot([x1+x_diffr], y1, 'ok', color='r', marker='^')
#    mean_a = mean(a)
#    sd_a = std(a)
#    n = 30
#    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
#                 mean_a + (sd_a / sqrt(n)) * t_val]
#    plt.plot([x_diffr-.05], mean_a, 'ok', markersize=20, marker='^')
#    plt.plot([x_diffr-.05, x_diffr-.05], mean_a_CI, '-k', linewidth=2)
    
    
    plt.title('DIFFERENCE IN PERCEIVED SPACING')
    x = [1, 1.5]
    labels = '-15','15', 'diff'
    plt.xticks(x, labels, rotation='horizontal')
    ax1.set_xlabel('')
    ax1.set_ylabel('Difference in Perceived Spacing (cm)')
    #plt.ylim(-20,25)


#   FIGURE 4.4A - difference in perceived sp -24 vs 24

    ax2 = fig.add_subplot(222)
    
     # PLOTTING -24
    x_diff = 1
    a = df['-24cm_grasp_vs_nograsp_sp']
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
    
    x_diffr = x_diff+.5
    
    # PLOTTING 24    
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
    
    plt.plot([.8,1.7], [0,0], '--k')    

    # Plotting difference
#    x_diffr = x_diff+1
#    a = df['-24_vs_24_sp']
#    for x1, y1, in zip(jit_a, a):
#        plt.plot([x1+x_diffr], y1, 'ok', color='r', marker='^')
#    mean_a = mean(a)
#    sd_a = std(a)
#    n = 30
#    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
#                 mean_a + (sd_a / sqrt(n)) * t_val]
#    plt.plot([x_diffr-.05], mean_a, 'ok', markersize=20, marker='^')
#    plt.plot([x_diffr-.05, x_diffr-.05], mean_a_CI, '-k', linewidth=2)
    
    
    plt.title('DIFFERENCE IN PERCEIVED SPACING')
    x = [1, 1.5]
    labels = '-24','24', 'diff'
    plt.xticks(x, labels, rotation='horizontal')
    ax2.set_xlabel('')
    ax2.set_ylabel('Difference in Perceived Spacing (cm)')
    #plt.ylim(-20,25)


#   FIGURE 4.3B - difference in perceived own -15 vs 15

    ax3 = fig.add_subplot(223)
    
     # PLOTTING -15
    x_diff = 1
    a = df['grasp_c15cm_own']
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
    
    x_diffr = x_diff+.5
    
    # PLOTTING 15    
    a = df['grasp_uc15cm_own']
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
    
    plt.plot([.8,1.7], [4,4], '--k')    

    # Plotting difference
#    x_diffr = x_diff+1
#    a = df['-15_vs_15_own']
#    for x1, y1, in zip(jit_a, a):
#        plt.plot([x1+x_diffr], y1, 'ok', color='r', marker='^')
#    mean_a = mean(a)
#    sd_a = std(a)
#    n = 30
#    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
#                 mean_a + (sd_a / sqrt(n)) * t_val]
#    plt.plot([x_diffr-.05], mean_a, 'ok', markersize=20, marker='^')
#    plt.plot([x_diffr-.05, x_diffr-.05], mean_a_CI, '-k', linewidth=2)
    
    
    plt.title('DIFFERENCE IN PERCEIVED OWNERSHIP')
    x = [1, 1.5]
    labels = '-15','15', 'diff'
    plt.xticks(x, labels, rotation='horizontal')
    y = [0,1,2,3,4,5,6,7,8]
    ylabels = ['','Str Disagree','Disagree','~ Disagree','0','~ Agree', 'Agree', 'Str Agree']
    plt.yticks(y, ylabels, rotation='horizontal')
    ax3.set_xlabel('Horizontal Locations')
    ax3.set_ylabel('Difference in Perceived Spacing (cm)')

#   FIGURE 4.4B - difference in perceived own -24 vs 24

    ax4 = fig.add_subplot(224)
    
     # PLOTTING -24
    x_diff = 1
    a = df['grasp_c24cm_own']
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
    
    x_diffr = x_diff+.5
    
    # PLOTTING 24    
    a = df['grasp_uc24cm_own']
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
    
    plt.plot([.8,1.7], [4,4], '--k')    

    # Plotting difference
#    x_diffr = x_diff+1
#    a = df['-24_vs_24_own']
#    for x1, y1, in zip(jit_a, a):
#        plt.plot([x1+x_diffr], y1, 'ok', color='r', marker='^')
#    mean_a = mean(a)
#    sd_a = std(a)
#    n = 30
#    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
#                 mean_a + (sd_a / sqrt(n)) * t_val]
#    plt.plot([x_diffr-.05], mean_a, 'ok', markersize=20, marker='^')
#    plt.plot([x_diffr-.05, x_diffr-.05], mean_a_CI, '-k', linewidth=2)
    
    
    plt.title('DIFFERENCE IN PERCEIVED OWNERSHIP')
    x = [1, 1.5]
    labels = '-24','24', 'diff'
    plt.xticks(x, labels, rotation='horizontal')
    y = [0,1,2,3,4,5,6,7,8]
    ylabels = ['','Str Disagree','Disagree','~ Disagree','0','~ Agree', 'Agree', 'Str Agree']
    plt.yticks(y, ylabels, rotation='horizontal')
    ax4.set_xlabel('Horizontal Locations')
    ax4.set_ylabel('')


def fig4_diff_sp(df):
    # plotting the differences in perceived spacing (grasp - no grasp) between
    # the different horizontal locations
    # Figure NOT ACTUALLY USED - just made to represent the differences
    
    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(111)
    
    # PLOTTING difference -15_vs_-24_sp
    x_diff = 1
    
    a = df['-15_vs_-24_sp']*-1
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
    
    
     # PLOTTING difference 15_vs_24_sp
    x_diff = 2
    
    a = df['15_vs_24_sp']*-1
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
    
    
    # PLOTTING difference -15_vs_15_sp
    x_diff = 3
    
    a = df['-15_vs_15_sp']*-1
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
    
    # PLOTTING difference -24_vs_24_sp
    x_diff = 4
    
    a = df['-24_vs_24_sp']*-1
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

    ax1.set_xlabel('Comparisons')
    ax1.set_ylabel('Difference in Perceived Spacing (cm)')
    x = [1, 2, 3, 4]
    labels = ['-15_vs_-24_sp','15_vs_24_sp','-15_vs_15_sp','-24_vs_24_sp']
    plt.xticks(x, labels, rotation='horizontal')
    plt.title('NG - G Diff for each location')




def fig4_diff_own(df):
    # plotting the differences in perceived ownership between
    # the different horizontal locations
    # Figure NOT ACTUALLY USED - just made to represent the differences
    
    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(111)
    
    # PLOTTING difference -15_vs_-24_sp
    x_diff = 1
    
    a = df['-15_vs_-24_own']
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
    
    
     # PLOTTING difference 15_vs_24_sp
    x_diff = 2
    
    a = df['15_vs_24_own']
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
    
    
    # PLOTTING difference -15_vs_15_sp
    x_diff = 3
    
    a = df['-15_vs_15_own']
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
    
    # PLOTTING difference -24_vs_24_sp
    x_diff = 4
    
    a = df['-24_vs_24_own']
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

    ax1.set_xlabel('Comparisons')
    ax1.set_ylabel('Difference in Perceived Spacing (cm)')
    x = [1, 2, 3, 4]
    labels = ['-15_vs_-24_sp','15_vs_24_sp','-15_vs_15_sp','-24_vs_24_sp']
    plt.xticks(x, labels, rotation='horizontal')
    plt.ylim(-7,7)
    plt.title('Diff in ownership for each locations')


