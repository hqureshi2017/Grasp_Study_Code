import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from scipy.stats import t
from numpy import max
from jitter import jitter
import numpy as np
from grasp_study_code import text_figure


def left_right_ng_g(df):
    
    # ------------------------------
    # SUBPLOT 211  R and L: NG and G
    # ------------------------------

    t_val = float(t.ppf([0.975], 30))

    fig = plt.figure(figsize=(9, 9))
    ax1 = fig.add_subplot(211)
    
    # Save data to text file
    text_figure('Fig3A + 3B',['nograsp_c15cm_l', 'grasp_c15cm_l',
                         'nograsp_c15cm_r', 'nograsp_c15cm_r',
                         '-15cm_grasp_vs_nograsp_l', '-15cm_grasp_vs_nograsp_r'], df)
    
    df.loc['sub01','nograsp_c15cm_l'] = np.nan 
    df.loc['sub01','grasp_c15cm_l'] = np.nan 
    df.loc['sub01','nograsp_c15cm_r'] = np.nan 
    df.loc['sub01','grasp_c15cm_r'] = np.nan 
    df.loc['sub01','-15cm_grasp_vs_nograsp_l'] = np.nan     
    df.loc['sub01','-15cm_grasp_vs_nograsp_r'] = np.nan

         
    # Plotting the Left Hand
    
    x_ng = 1
    x_g = 2

    dat = []
    a = list(df['nograsp_c15cm_l'])
    b = list(df['grasp_c15cm_l'])
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = jitter(dat)
    for x1, x2, y1, y2 in zip(jit_a, jit_b, a, b):
        plt.plot([x1 + x_ng, x2 + x_g], [y1, y2], '-', color='lightgrey')

    mean_a = np.nanmean(a)
    sd_a = np.nanstd(a)
    n = 29
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot(x_ng, mean_a, 'ok', markersize=12)
    plt.plot([x_ng, x_ng], mean_a_CI, '-k', linewidth=2)

    mean_b = np.nanmean(b)
    sd_b = np.nanstd(b)
    n = 29
    mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
                 mean_b + (sd_b / sqrt(n)) * t_val]
    x_val = x_g + max(jit_b) + .01
    plt.plot([x_val, x_val], mean_b_CI, '-k', linewidth=2)
    plt.plot(x_val, mean_b, 'ok', markersize=12,color='w', mew=2)
    
    # Plotting the Right Hand    
    x_ng = 3.5
    x_g = 4.5

    dat = []
    a = list(df['nograsp_c15cm_r'])
    b = list(df['grasp_c15cm_r'])
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = jitter(dat)
    for x1, x2, y1, y2 in zip(jit_a, jit_b, a, b):
        plt.plot([x1 + x_ng, x2 + x_g], [y1, y2], '-', color='lightgrey')

    mean_a = np.nanmean(a)
    sd_a = np.nanstd(a)
    n = 29
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot(x_ng, mean_a, 'ok', markersize=12)
    plt.plot([x_ng, x_ng], mean_a_CI, '-k', linewidth=2)

    mean_b = np.nanmean(b)
    sd_b = np.nanstd(b)
    n = 29
    mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
                 mean_b + (sd_b / sqrt(n)) * t_val]
    x_val = x_g + max(jit_b) + .01
    plt.plot([x_val, x_val], mean_b_CI, '-k', linewidth=2)
    plt.plot(x_val, mean_b, 'ok', markersize=12, color='w', mew=2)
    
    #Removing ticks from axes
    ax1.tick_params( axis='x', which='minor', direction='out', length=60, top='off', right='off', labelbottom = 'on' )
    ax1.tick_params( axis='x', which='major', bottom='off', top='off', labelbottom='on', left='off', right='off', labelsize=14)
    ax1.tick_params( axis='y', which='both', direction='out', length=3, bottom='off', top='off', labelbottom='on', right='off', left='on', labelsize=14)

    # Hide the right and top spines
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    plt.xlim([0.5, 5.5])
    
    plt.title('')
    xticks = [1, 1.5, 2, 3.5, 4, 4.5 ]
    labels = ['no grasp','left index','grasp', 'no grasp','right index','grasp']
    plt.xticks(xticks, labels)
    
    # Vertical alignment of xtick labels
    va = [ 0, -.05, 0, 0, -.05, 0]
    for z, y in zip( ax1.get_xticklabels( ), va ):
        z.set_y( y )
    
    
    ax1.set_xlabel('')
    
    
    ax1.set_ylabel('Perceived location (cm)',size=20)
    plt.tight_layout()


    
    # -------------------------------
    # SUBPLOT 212  R and L: NG-G diff 
    # -------------------------------
    ax2 = fig.add_subplot(212)
    x_diff = 1
    dat = []
    a = list(df['-15cm_grasp_vs_nograsp_l'])
    b = list(df['-15cm_grasp_vs_nograsp_l'])
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = jitter(dat)
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1*3+x_diff], y1, 'ok', color='k',marker='^')

    mean_a = np.nanmean(a)
    sd_a = np.nanstd(a)
    n = 29
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diff-.05], mean_a, 'ok', markersize=10, marker='^')
    plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)
    
    x_diffr = 1.5
    # PLOTTING difference in location -15cm RIGHT
    
    dat = []
    a = list(df['-15cm_grasp_vs_nograsp_r'])
    b = list(df['-15cm_grasp_vs_nograsp_r'])
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = jitter(dat)
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1*3+x_diffr], y1, 'ok', color='k', marker='^')

    mean_a = np.nanmean(a)
    sd_a = np.nanstd(a)
    n = 29
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diffr-.05], mean_a, 'ok', markersize=10, marker='^')
    plt.plot([x_diffr-.05, x_diffr-.05], mean_a_CI, '-k', linewidth=2)
    
    yticks = [-10, -7.5, -5, -2.5, 0, 2.5,5, 7.5,10]
    plt.yticks(yticks)
    plt.title('')
    
    #Removing ticks from axes
    ax2.tick_params( axis='x', which='minor', direction='out', length=60, top='off', right='off', labelbottom = 'on' )
    ax2.tick_params( axis='x', which='major', bottom='off', top='off', labelbottom='on', left='off', right='off', labelsize=14)
    ax2.tick_params( axis='y', which='both', direction='out', length=3, bottom='off', top='off', labelbottom='on', right='off', left='on', labelsize=14)

    # Hide the right and top spines
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
   
    plt.xlim([0.7, 2])
    plt.ylim([-10, 10])
    
   
    plt.plot([0.5,2], [0,0], '-k', alpha=0.5)

    x = [1, 1.5]
    labels = ['left index', 'right index']
    
    plt.xticks(x, labels, rotation='horizontal')
    ax2.set_ylabel('Difference perceived location (cm)',size=20)
    plt.tight_layout()
    
    
    plt.savefig( './Graphs/data/' + 'fig_crossed_15cm_L_R.pdf')  
    plt.savefig( './Graphs/data/' + 'fig_crossed_15cm_L_R.svg')  
