########### HASSAN ##############
#   File contains code for plotting the Ownership measure (fig 3)
#   
################################## 

import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from scipy.stats import t
from numpy import mean
from numpy import std
from jitter import jitter
from grasp_study_code import text_figure

def own_outcomes(df):
    # FIGURE 4
    jit = 0.01
    
    # ------------------------------
    # SUBPLOT 211  OWNERSHIP
    # ------------------------------

    t_val = float(t.ppf([0.975], 30))

    fig = plt.figure(figsize=(9, 9))
    ax1 = fig.add_subplot(211)
    
    # Save data to text file
    text_figure('Fig4A + 4B',['grasp_c24cm_own',
                        'grasp_c15cm_own',
                        'grasp_uc24cm_own',
                        'grasp_uc15cm_own',
                        '-15_vs_-24_own',
                         '15_vs_24_own',
                         '-15_vs_15_own',
                         '-24_vs_24_own',], df)
    
     # PLOTTING -24cm
    x_own = 1

    dat = []
    a = list(df['grasp_c24cm_own'])
    b = list(df['grasp_c24cm_own'])
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = jitter(dat, jit)
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1*2+x_own], y1, 'ok', color='k', markersize=3)

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_own-.05], mean_a, 'ok', markersize=10)
    plt.plot([x_own-.05, x_own-.05], mean_a_CI, '-k', linewidth=2)
    
     # PLOTTING -15cm
    x_own = x_own+.8
    
    dat = []
    a = list(df['grasp_c15cm_own'])
    b = list(df['grasp_c15cm_own'])
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = jitter(dat, jit)
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1*2+x_own], y1, 'ok', color='k', markersize=3)

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_own-.05], mean_a, 'ok', markersize=10)
    plt.plot([x_own-.05, x_own-.05], mean_a_CI, '-k', linewidth=2)
    
     # PLOTTING 15cm
    x_own = x_own+.8
    
    dat = []
    a = list(df['grasp_uc15cm_own'])
    b = list(df['grasp_uc15cm_own'])
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = jitter(dat, jit)
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1*2+x_own], y1, 'ok', color='k', markersize=3)

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_own-.05], mean_a, 'ok', markersize=10)
    plt.plot([x_own-.05, x_own-.05], mean_a_CI, '-k', linewidth=2)
    
     # PLOTTING 24cm
    x_own = x_own+.8
    
    dat = []
    a = list(df['grasp_uc24cm_own'])
    b = list(df['grasp_uc24cm_own'])
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = jitter(dat, jit)
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1*2+x_own], y1, 'ok', color='k', markersize=3)

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_own-.05], mean_a, 'ok', markersize=10)
    plt.plot([x_own-.05, x_own-.05], mean_a_CI, '-k', linewidth=2)
    

    #Removing ticks from axes
    ax1.tick_params( axis='x', which='minor', direction='out', length=60, top='off', right='off', labelbottom = 'on' )
    ax1.tick_params( axis='x', which='major', bottom='off', top='off', labelbottom='on', left='off', right='off',labelsize=14)
    ax1.tick_params( axis='y', which='both', bottom='off', top='off', direction='out', length=3, labelbottom='on', left='on', right='off',labelsize=14)

    # Hide the right and top spines
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    
            
    ax1.set_xlabel('')
    ax1.set_ylabel('Perceived ownership', size=20)
    xticks = [1,1.4, 1.8, 2.6, 3, 3.4]
    labels = ['-24 cm', 'hands crossed', '-15 cm',
              '15 cm', 'hands uncrossed','24 cm']
    plt.xticks(xticks, labels, rotation='horizontal')
    
     # vertical alignment of xtick labels
    va = [ 0, -.05, 0, 0, -.05, 0, 
          0, -.05, 0, 0, -.05, 0, ]
    for z, y in zip( ax1.get_xticklabels( ), va ):
        z.set_y( y )
        
    y = [0,1,2,3,4,5,6,7,8]
    ylabels = ['','Str disagree','Disagree','Sm disagree','Neutral','Sm agree', 'Agree', 'Str agree']
    plt.yticks(y, ylabels, rotation='horizontal')
    plt.title('')
    
    
    # ----------------------------------------------------------
    # SUBPLOT 212  PERCEIVED OWNERSHIP BETWEEN PAIRED COMPARISONS
    # ----------------------------------------------------------
    ax2 = fig.add_subplot(212)
            
    # PLOTTING difference -15cm vs -24cm
    x_diff = 1
    
    dat = []
    a = list(df['-15_vs_-24_own'])
    b = list(df['-15_vs_-24_own'])
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = jitter(dat)
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1*3+x_diff], y1, 'ok', marker='^')
    
    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diff-.05], mean_a, 'ok', markersize=10, marker='^')
    plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)
    
    
     # PLOTTING difference 15 vs 24cm
    x_diff = 1.8
    
    dat = []
    a = list(df['15_vs_24_own'])
    b = list(df['15_vs_24_own'])
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = jitter(dat)
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1*3+x_diff], y1, 'ok', marker='^')
    
    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diff-.05], mean_a, 'ok', markersize=10, marker='^')
    plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)
    
    
    # PLOTTING difference -15 vs 15 cm
    x_diff = 2.6
    
    
    dat = []
    a = list(df['-15_vs_15_own'])
    b = list(df['-15_vs_15_own'])
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = jitter(dat)
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1*3+x_diff], y1, 'ok', marker='^')
    
    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diff-.05], mean_a, 'ok', markersize=10, marker='^')
    plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)
    
    # PLOTTING difference -24_vs_24_sp
    x_diff = 3.4
    
    
    dat = []
    a = list(df['-24_vs_24_own'])
    b = list(df['-24_vs_24_own'])
    dat = [pd.Series(a), pd.Series(b)]    
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1*3+x_diff], y1, 'ok', marker='^')
    
    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diff-.05], mean_a, 'ok', markersize=10, marker='^')
    plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)
    
    plt.plot([3.8,.8], [0,0], '-k', alpha=0.5)

    ax2.grid( 'off', axis='x' )
    ax2.grid( 'off', axis='y' )
    
    # Hide the right and top spines
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    #Removing ticks from axes
    ax2.tick_params( axis='x', which='minor', direction='out', length=60, top='off', right='off', labelbottom = 'on' )
    ax2.tick_params( axis='x', which='major', bottom='off', top='off', labelbottom='on', left='off', right='off',labelsize=14)
    ax2.tick_params( axis='y', which='both', bottom='off', top='off', direction='out', length=3, labelbottom='on', left='on', right='off', labelsize=14)
    
    ax2.set_xlabel('')
    ax2.set_ylabel('Difference in perceived ownership', size=20)
    
    xticks = [1,1.4, 1.8, 2.6, 3, 3.4]
    labels = ['-15cm vs -24cm', 'distance', '15cm vs 24cm',
              '-15cm vs 15cm', 'midline','-24cm vs 24cm']
    plt.xticks(xticks, labels)
    
    plt.ylim(-4.5,4.5)
    yticks = [-4,-3,-2,-1,0, 1,2,3,4]
    plt.yticks(yticks)

    # vertical alignment of xtick labels
    va = [ 0, -.05, 0, 0, -.05, 0, 
          0, -.05, 0, 0, -.05, 0, ]
    for z, y in zip( ax2.get_xticklabels( ), va ):
        z.set_y( y )
    
    plt.title('')
    
    
    
    plt.savefig( './Graphs/data/' + 'fig_4.pdf')   
    plt.savefig( './Graphs/data/' + 'fig_4.svg')

    
    
    
    
    
    
    
    
    
    
def own_diff_outcomes(df):
    # This GRAPH is not actually used. But is a nice way to see the differences
    
    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(111)
    
    t_val = float(t.ppf([0.975], 30))
    
     # PLOTTING difference -15cm vs -24cm
    x_diff = 1
    
    dat = []
    a = list(df['-15_vs_-24_own'])
    b = list(df['-15_vs_-24_own'])
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = jitter(dat, jit)
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

    dat = []
    a = list(df['15_vs_24_own'])
    b = list(df['15_vs_24_own'])
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = jitter(dat, jit)
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
    

    dat = []
    a = list(df['-15_vs_15_own'])
    b = list(df['-15_vs_15_own'])
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = jitter(dat, jit)
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
    

    dat = []
    a = list(df['-24_vs_24_own'])
    b = list(df['-24_vs_24_own'])
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = jitter(dat, jit)
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
    ax1.set_ylabel('Difference in perceived ownership (cm)')
    x = [1, 2, 3, 4]
    labels = ['-15_vs_-24','15_vs_24','-15_vs_15','-24_vs_24']
    plt.xticks(x, labels, rotation='horizontal')
    
    plt.title('OWNERSHIP_diff')