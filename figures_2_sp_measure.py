########### HASSAN ##############
#   File contains code for plotting the Spacing measure (fig 2)
#   It contains two functions: 1) Plots the Perceived Spacing for Grasp and No
#   Grasp across all 4 locations.
#   2) Plots the difference between the Grasp and No Grasp for each location
#   
################################## 

import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from scipy.stats import t
from numpy import max
from numpy import mean
from numpy import std
from jitter import jitter
from grasp_study_code import text_figure

def sp_outcomes(df):
    #    Function that plots the difference between Grasp and No Grasp across all 
    #    trials
    
    t_val = float(t.ppf([0.975], 30))

    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(111)
    
    # Save data to text file
    text_figure('Fig2A',['nograsp_c24cm_sp', 'grasp_c24cm_sp',
                         'nograsp_c15cm_sp', 'grasp_c15cm_sp',
                         'nograsp_uc24cm_sp', 'grasp_uc24cm_sp',
                         'nograsp_uc15cm_sp', 'grasp_uc15cm_sp'], df)
    # PLOTTING SPACING -24cm
    x_ng = 0.8
    x_g = 1.2
    
    dat = []
    a = list(df['nograsp_c24cm_sp'])
    b = list(df['grasp_c24cm_sp'])
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = jitter(dat)
    for x1, x2, y1, y2 in zip(jit_a, jit_b, a, b):
        plt.plot([x1 + x_ng, x2 + x_g], [y1, y2], '-', color='lightgrey')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_ng, x_ng], mean_a_CI, '-k', linewidth=2)
    plt.plot(x_ng, mean_a, 'ok', markersize=12)

    mean_b = mean(b)
    sd_b = std(b)
    n = 30
    mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
                 mean_b + (sd_b / sqrt(n)) * t_val]

    x_val = x_g + max(jit_b) + 0.02
    plt.plot([x_val, x_val], mean_b_CI, '-k', linewidth=2)
    plt.plot(x_val, mean_b, 'ok', markersize=12, color='w', mew=2)
    

    # PLOTTING SPACING -15cm
    x_ng = 1.8
    x_g = 2.2

    dat = []
    a = list(df['nograsp_c15cm_sp'])
    b = list(df['grasp_c15cm_sp'])
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = jitter(dat)
    for x1, x2, y1, y2 in zip(jit_a, jit_b, a, b):
        plt.plot([x1 + x_ng, x2 + x_g], [y1, y2], '-', color='lightgrey')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_ng, x_ng], mean_a_CI, '-k', linewidth=2)
    plt.plot(x_ng, mean_a, 'ok', markersize=12)

    mean_b = mean(b)
    sd_b = std(b)
    n = 30
    mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
                 mean_b + (sd_b / sqrt(n)) * t_val]

    x_val = x_g + max(jit_b) + 0.02
    plt.plot([x_val, x_val], mean_b_CI, '-k', linewidth=2)
    plt.plot(x_val, mean_b, 'ok', markersize=12,color='w', mew=2)

    # PLOTTING SPACING 15cm
    x_ng = 2.8
    x_g = 3.2

    dat = []
    a = list(df['nograsp_uc15cm_sp'])
    b = list(df['grasp_uc15cm_sp'])
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = jitter(dat)
    for x1, x2, y1, y2 in zip(jit_a, jit_b, a, b):
        plt.plot([x1 + x_ng, x2 + x_g], [y1, y2], '-', color='lightgrey')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_ng, x_ng], mean_a_CI, '-k', linewidth=2)
    plt.plot(x_ng, mean_a, 'ok', markersize=12)

    mean_b = mean(b)
    sd_b = std(b)
    n = 30
    mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
                 mean_b + (sd_b / sqrt(n)) * t_val]

    x_val = x_g + max(jit_b) + 0.02
    plt.plot([x_val, x_val], mean_b_CI, '-k', linewidth=2)
    plt.plot(x_val, mean_b, 'ok', markersize=12, color='w', mew=2)


    # PLOTTING SPACING 24cm
    x_ng = 3.8
    x_g = 4.2

    dat = []
    a = list(df['nograsp_uc24cm_sp'])
    b = list(df['grasp_uc24cm_sp'])
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = jitter(dat)
    for x1, x2, y1, y2 in zip(jit_a, jit_b, a, b):
        plt.plot([x1 + x_ng, x2 + x_g], [y1, y2], '-', color='lightgrey')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_ng, x_ng], mean_a_CI, '-k', linewidth=2)
    plt.plot(x_ng, mean_a, 'ok', markersize=12)

    mean_b = mean(b)
    sd_b = std(b)
    n = 30
    mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
                 mean_b + (sd_b / sqrt(n)) * t_val]

    x_val = x_g + max(jit_b) + 0.02
    plt.plot([x_val, x_val], mean_b_CI, '-k', linewidth=2)
    plt.plot(x_val, mean_b, 'ok', markersize=12, color='w', mew=2)
    ax1.set_xlabel('', size=20)
    ax1.set_ylabel('Perceived spacing (cm)',size=20)
    
    xticks = [.8, 1, 1.2, 1.55, 1.8, 2, 2.2, 
              2.8, 3, 3.2, 3.55, 3.8, 4, 4.2, ]
    labels = ['no grasp','-24 cm','grasp', 'hands crossed', 'no grasp','-15 cm','grasp',
              'no grasp','15 cm','grasp', 'hands uncrossed', 'no grasp','24 cm','grasp']
    plt.xticks(xticks, labels)

    ax1.grid( 'off', axis='x' )
    ax1.grid( 'off', axis='y' )
    
    # Hide the right and top spines
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
        
    #Removing ticks from axes
    ax1.tick_params( axis='x', which='major', bottom='off', top='off', labelbottom='on', left='off', right='off', labelsize=14)
    ax1.tick_params( axis='y', which='major', direction='out', length=3, bottom='off', top='off', labelbottom='on', right='off', labelsize=14)

        # vertical alignment of xtick labels
    va = [ 0, -.05, 0, -.1, 0, -.05, 0, 
          0, -.05, 0, -.1, 0, -.05, 0]
    for z, y in zip( ax1.get_xticklabels( ), va ):
        z.set_y( y )
    plt.tight_layout
    plt.title('')
    plt.savefig( './Graphs/data/' + 'fig_2A.pdf')
    plt.savefig( './Graphs/data/' + 'fig_2A.svg')


    # FIGURE 2B - Difference
       
def sp_diff_outcomes(df):
    
    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(111)
    
    t_val = float(t.ppf([0.975], 30))
    
    # Write data to text file
    text_figure('Fig2B',['-24cm_grasp_vs_nograsp_sp',
                         '-15cm_grasp_vs_nograsp_sp',
                         '24cm_grasp_vs_nograsp_sp',
                         '15cm_grasp_vs_nograsp_sp',], df)
    # PLOTTING difference -24cm
    x_diff = 1

    dat = []
    a = list(df['-24cm_grasp_vs_nograsp_sp'])
    b = list(df['-24cm_grasp_vs_nograsp_sp'])
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = jitter(dat)
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1*3+x_diff], y1, 'ok', marker='^')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diff-.1], mean_a, 'ok',markersize=12, marker='^')
    plt.plot([x_diff-.1, x_diff-.1], mean_a_CI, '-k', linewidth=2)
    
    
     # PLOTTING difference -15cm
    x_diff = 1.5

    dat = []
    a = list(df['-15cm_grasp_vs_nograsp_sp'])
    b = list(df['-15cm_grasp_vs_nograsp_sp'])
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = jitter(dat)
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1*3+x_diff], y1, 'ok', marker='^')
    
    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diff-.1], mean_a, 'ok', markersize=12, marker='^')
    plt.plot([x_diff-.1, x_diff-.1], mean_a_CI, '-k', linewidth=2)
    
    
    # PLOTTING difference 15cm
    x_diff = 2
    

    dat = []
    a = list(df['15cm_grasp_vs_nograsp_sp'])
    b = list(df['15cm_grasp_vs_nograsp_sp'])
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = jitter(dat)
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1*3+x_diff], y1, 'ok', marker='^')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diff-.1], mean_a, 'ok', markersize=12, marker='^')
    plt.plot([x_diff-.1, x_diff-.1], mean_a_CI, '-k', linewidth=2)
    
    # PLOTTING difference 24cm
    x_diff = 2.5
    

    dat = []
    a = list(df['24cm_grasp_vs_nograsp_sp'])
    b = list(df['24cm_grasp_vs_nograsp_sp'])
    dat = [pd.Series(a), pd.Series(b)]    
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1*3+x_diff], y1, 'ok', marker='^')

    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diff-.1], mean_a, 'ok', markersize=12, marker='^')
    plt.plot([x_diff-.1, x_diff-.1], mean_a_CI, '-k', linewidth=2)
    
   
    plt.plot([2.7,.8], [0,0], '-k', alpha=0.5)

    ax1.grid( 'off', axis='x' )
    ax1.grid( 'off', axis='y' )
    
    # Hide the right and top spines
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    
    #Removing ticks from axes
    #ax1.tick_params( axis='x', which='minor', direction='out', length=60, top='off', right='off', labelbottom = 'on' )
    ax1.tick_params( axis='x', which='major', bottom='off', top='off', labelbottom='on', left='off', right='off',labelsize=14)
    ax1.tick_params( axis='y', which='both', bottom='off', direction='out', length=3, top='off', labelbottom='on', left='on', right='off', labelsize=14)

    ax1.set_xlabel('')
    ax1.set_ylabel('Difference in perceived spacing (cm)',size=20)
    
    xticks = [1, 1.25, 1.5, 2, 2.25, 2.5 ]
    labels = ['- 24 cm', 'hands crossed', '- 15 cm',
              '15 cm', 'hands uncrossed','24 cm']
    plt.xticks(xticks, labels)

    # vertical alignment of xtick labels
    va = [ 0, -.05, 0, 0, -.05, 0, 
          0, -.05, 0, 0, -.05, 0, ]
    for z, y in zip( ax1.get_xticklabels( ), va ):
        z.set_y( y )
    
    plt.title('')
    
    plt.savefig( './Graphs/data/' + 'fig_2B.pdf')   
    plt.savefig( './Graphs/data/' + 'fig_2B.svg')
