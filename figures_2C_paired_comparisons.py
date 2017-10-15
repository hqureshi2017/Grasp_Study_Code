
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from scipy.stats import t
from numpy import max
from numpy import mean
from numpy import std
from jitter import jitter
from grasp_study_code import text_figure


def paired(df):
    # PERCEIVED SPACING BETWEEN PAIRED COMPARISONS
    # Following discussion with MH, AB and HQ
    
    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(111)
    
    t_val = float(t.ppf([0.975], 30))
    
    # Write data to text file
    text_figure('Fig2C: Spacing Comparisons',['-15_vs_-24_sp',
                         '15_vs_24_sp',
                         '-15_vs_15_sp',
                         '-24_vs_24_sp',], df)
    # PLOTTING difference -15cm vs -24cm
    x_diff = 1
    
    dat = []
    a = list(df['-15_vs_-24_sp'])
    b = list(df['-15_vs_-24_sp'])
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = jitter(dat)
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1*3+x_diff], y1, 'ok', marker='^')
    
    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diff-.05], mean_a, 'ok', markersize=12, marker='^')
    plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)
    
    
     # PLOTTING difference 15 vs 24cm
    x_diff = 1.5
    
    dat = []
    a = list(df['15_vs_24_sp'])
    b = list(df['15_vs_24_sp'])
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = jitter(dat)
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1*3+x_diff], y1, 'ok', marker='^')
    
    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diff-.05], mean_a, 'ok', markersize=12, marker='^')
    plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)
    
    
    # PLOTTING difference -15 vs 15 cm
    x_diff = 2
    
    
    dat = []
    a = list(df['-15_vs_15_sp'])
    b = list(df['-15_vs_15_sp'])
    dat = [pd.Series(a), pd.Series(b)]
    jit_a, jit_b = jitter(dat)
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1*3+x_diff], y1, 'ok', marker='^')
    
    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diff-.05], mean_a, 'ok', markersize=12, marker='^')
    plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)
    
    # PLOTTING difference -24_vs_24_sp
    x_diff = 2.5
    
    
    dat = []
    a = list(df['-24_vs_24_sp'])
    b = list(df['-24_vs_24_sp'])
    dat = [pd.Series(a), pd.Series(b)]    
    for x1, y1, in zip(jit_a, a):
        plt.plot([x1*3+x_diff], y1, 'ok', marker='^')
    
    mean_a = mean(a)
    sd_a = std(a)
    n = 30
    mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
                 mean_a + (sd_a / sqrt(n)) * t_val]
    plt.plot([x_diff-.05], mean_a, 'ok', markersize=12, marker='^')
    plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)
    
    plt.plot([2.7,.8], [0,0], '-k', alpha=0.5)

    ax1.grid( 'off', axis='x' )
    ax1.grid( 'off', axis='y' )
    
    # Hide the right and top spines
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    
    #Removing ticks from axes
    ax1.tick_params( axis='x', which='minor', direction='out', length=60, top='off', right='off', labelbottom = 'on' )
    ax1.tick_params( axis='x', which='major', bottom='off', top='off', labelbottom='on', left='off', right='off', labelsize=14)
    ax1.tick_params( axis='y', which='both', bottom='off', top='off',direction='out', length=3, labelbottom='on', left='on', right='off', labelsize=14)
    
    ax1.set_xlabel('')
    ax1.set_ylabel('Difference in perceived spacing (cm)',size=20)
    
    xticks = [1, 1.25, 1.5, 2, 2.25, 2.5 ]
    labels = ['-15cm vs -24cm', 'distance', '15cm vs 24cm',
              '-15cm vs 15cm', 'midline','-24cm vs 24cm']
    plt.xticks(xticks, labels)
    
    # vertical alignment of xtick labels
    va = [ 0, -.05, 0, 0, -.05, 0, 
          0, -.05, 0, 0, -.05, 0, ]
    for z, y in zip( ax1.get_xticklabels( ), va ):
        z.set_y( y )
    
    plt.title('')
    
    plt.savefig( './Graphs/data/' + 'paired_comp.pdf')   
    plt.savefig( './Graphs/data/' + 'paired_comp.svg')   
  