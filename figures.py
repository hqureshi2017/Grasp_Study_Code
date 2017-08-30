

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


########### HASSAN ##############
#   The code contains the graphs for data inspection and the actual graphs used for study
#   I haven't been able to get the 'function' to work always, but the graphs all seem to 
#   work for me when I specifically make one at a time.
################################## 



def _jitter(data, jit=None):
    if jit is None:
        jit = 0.005

    a = data[0]
    b = data[1]

    duplicate_a = []
    duplicate_b = []
    jitter_a = [''] * len(a)
    jitter_b = [''] * len(b)
    jitter_a[0] = 0
    jitter_b[0] = 0

    for i in np.arange(1, len(a), 1):
        a_val = a.values[i]
        b_val = b.values[i]
        if a_val in a.values[0:i]:
            duplicate_a.append(a_val)
            val = jit * duplicate_a.count(a_val)
            jitter_a[i] = val
        else:
            jitter_a[i] = 0

        if b_val in b.values[0:i]:
            duplicate_b.append(b_val)
            val = jit * duplicate_b.count(b_val)
            jitter_b[i] = val
        else:
            jitter_b[i] = 0
    return jitter_a, jitter_b

#---------------------------
# DATA CHECKING
# --------------------------
def sp_own_data_check(df):
#    Function that plots the absolute values for spacing across the four trials.
#    Allows to check for any bad data points. 
#    
    
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2})

    fig = plt.figure(figsize=(15, 9))

    ax1 = fig.add_subplot(221)  # NO-GRASP SPACING

    dat = df[['nograsp_uc24cm_sp', 'nograsp_uc15cm_sp', 'nograsp_c24cm_sp', 'nograsp_c15cm_sp']].copy()
    dat['nograsp_c24cm_sp'] = dat['nograsp_c24cm_sp'] * -1
    dat['nograsp_c15cm_sp'] = dat['nograsp_c15cm_sp'] * -1

    sns.swarmplot(data=dat)
    ax1.set_xlabel('')
    ax1.set_ylabel('perceived spacing (cm)')
    x = [0, 1, 2, 3]
    labels = '24cm', '15cm', '-24cm', '-15cm'
    plt.xticks(x, labels, rotation='horizontal')
    plt.title('NO GRASP')
    plt.ylim(0,45)
    ax2 = fig.add_subplot(222)  # GRASP SPACING

    dat = df[['grasp_uc24cm_sp', 'grasp_uc15cm_sp', 'grasp_c24cm_sp', 'grasp_c15cm_sp']].copy()
    dat['grasp_c24cm_sp'] = dat['grasp_c24cm_sp'] * -1
    dat['grasp_c15cm_sp'] = dat['grasp_c15cm_sp'] * -1

    sns.swarmplot(data=dat)
    ax2.set_xlabel('Horizontal Locations')
    ax2.set_ylabel('')
    x = [0, 1, 2, 3]
    labels = '24cm', '15cm', '-24cm', '-15cm'
    plt.xticks(x, labels, rotation='horizontal')
    plt.title('GRASP')
    plt.ylim(0,45)
    
    ax3 = fig.add_subplot(223)  # OWNERSHIP

    dat = df[['grasp_uc24cm_own', 'grasp_uc15cm_own', 'grasp_c24cm_own', 'grasp_c15cm_own']].copy()
    sns.swarmplot(data=dat)
    ax3.set_xlabel('Horizontal Locations')
    ax3.set_ylabel('Perceived Ownership')
    x = [0, 1, 2, 3]
    labels = '24cm', '15cm', '-24cm', '-15cm'
    plt.xticks(x, labels, rotation='horizontal')
    plt.title('OWNERSHIP')
    
def location_data_check(df):
    fig = plt.figure(figsize=(15, 9))
        
    ax1 = fig.add_subplot(222)  # Location_Left_Grasp
    dat = df[['grasp_uc24cm_l', 'grasp_uc15cm_l', 'grasp_c24cm_l', 'grasp_c15cm_l']].copy()
    dat['grasp_c24cm_l'] = dat['grasp_c24cm_l'] -36
    dat['grasp_c15cm_l'] = dat['grasp_c15cm_l'] -36
    dat['grasp_uc24cm_l'] = dat['grasp_uc24cm_l'] -36
    dat['grasp_uc24cm_l'] = dat['grasp_uc24cm_l'] *-1
    dat['grasp_uc15cm_l'] = dat['grasp_uc15cm_l'] -36
    dat['grasp_uc15cm_l'] = dat['grasp_uc15cm_l'] *-1
    sns.swarmplot(data=dat)
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    x = [0, 1, 2,3]
    labels = '24cm', '15cm', '-24cm', '-15cm'
    plt.xticks(x, labels, rotation='horizontal')
    plt.title('2')
    plt.ylim(0,35)
    
        
    ax2 = fig.add_subplot(221)  # Location_Left_No_Grasp
    
    dat = df[['nograsp_uc24cm_l', 'nograsp_uc15cm_l', 'nograsp_c24cm_l', 'nograsp_c15cm_l']].copy()
    dat['nograsp_c24cm_l'] = dat['nograsp_c24cm_l'] -36
    dat['nograsp_c15cm_l'] = dat['nograsp_c15cm_l'] -36
    dat['nograsp_uc24cm_l'] = dat['nograsp_uc24cm_l'] -36
    dat['nograsp_uc24cm_l'] = dat['nograsp_uc24cm_l'] *-1
    dat['nograsp_uc15cm_l'] = dat['nograsp_uc15cm_l'] -36
    dat['nograsp_uc15cm_l'] = dat['nograsp_uc15cm_l'] *-1
    
    sns.swarmplot(data=dat)
    ax2.set_xlabel('')
    ax2.set_ylabel('Perceived location (cm)')
    x = [0, 1, 2,3]
    plt.xticks(x, labels, rotation='horizontal')
    plt.title('1')
    plt.ylim(0,35)

    ax3 = fig.add_subplot(224)  # Location_Right_Grasp
    dat = df[['grasp_uc24cm_r', 'grasp_uc15cm_r', 'grasp_c24cm_r', 'grasp_c15cm_r']].copy()
    dat['grasp_c24cm_r'] = dat['grasp_c24cm_r'] -36
    dat['grasp_c15cm_r'] = dat['grasp_c15cm_r'] -36
    dat['grasp_uc24cm_r'] = dat['grasp_uc24cm_r'] -36
    #dat['grasp_uc24cm_r'] = dat['grasp_uc24cm_r'] *-1
    dat['grasp_uc15cm_r'] = dat['grasp_uc15cm_r'] -36
    #dat['grasp_uc15cm_r'] = dat['grasp_uc15cm_r'] *-1
    
    sns.swarmplot(data=dat)
    ax3.set_xlabel('Horizontal Locations')
    ax1.set_ylabel('')
    x = [0, 1, 2,3]
    plt.xticks(x, labels, rotation='horizontal')
    plt.title('4')
    plt.ylim(-20,15)
        
    ax4 = fig.add_subplot(223)  # Location_Right_No_Grasp
    dat = df[['nograsp_uc24cm_r', 'nograsp_uc15cm_r', 'nograsp_c24cm_r', 'nograsp_c15cm_r']].copy()
    dat['nograsp_c24cm_r'] = dat['nograsp_c24cm_r'] -36
    dat['nograsp_c15cm_r'] = dat['nograsp_c15cm_r'] -36
    dat['nograsp_uc24cm_r'] = dat['nograsp_uc24cm_r'] -36
    #dat['nograsp_uc24cm_r'] = dat['nograsp_uc24cm_r'] *-1
    dat['nograsp_uc15cm_r'] = dat['nograsp_uc15cm_r'] -36
    #dat['nograsp_uc15cm_r'] = dat['nograsp_uc15cm_r'] *-1
    
    sns.swarmplot(data=dat)
    ax4.set_xlabel('Horizontal Locations')
    ax4.set_ylabel('Perceived location (cm)')
    x = [0, 1, 2,3]
    plt.xticks(x, labels, rotation='horizontal')
    plt.title('3')
    plt.ylim(-20,15)
    
def validation_check(df):
    
    fig = plt.figure(figsize=(15,9))
    
    trials= ['L24_1', 'L24_2', 'L15_1', 'L15_2', 
              'R15_1', 'R15_2', 'R24_1', 'R24_2']
    #if blinding:
    #    random.shuffle(trials)
        
    ax1 = fig.add_subplot(221)  # Validation_spacing
    dat = df[['L24_1', 'L24_2', 'L15_1', 'L15_2', 
              'R15_1', 'R15_2', 'R24_1', 'R24_2']].copy()
    dat['R15_1'] = dat['R15_1'] * -1
    dat['R15_2'] = dat['R15_2'] * -1
    dat['R24_1'] = dat['R24_1'] * -1
    dat['R24_2'] = dat['R24_2'] * -1
    
    sns.swarmplot(data=dat)
    ax1.set_xlabel('')
    ax1.set_ylabel('Perceived Spacing (cm)')
    x = [0, 1, 2,3,4,5,6,7]
    labels = trials
    plt.xticks(x, labels, rotation='horizontal')
    plt.title('A')
    
    ax2 = fig.add_subplot(224)  # Validation_location_R
    dat = df[['loc_R_L24_1', 'loc_R_L24_2', 'loc_R_L15_1', 'loc_R_L15_2', 
              'loc_R_R15_1', 'loc_R_R15_2', 'loc_R_R24_1', 'loc_R_R24_2']].copy()
    dat['loc_R_L24_1'] = dat['loc_R_L24_1'] -36
    dat['loc_R_L24_2'] = dat['loc_R_L24_2'] -36
    dat['loc_R_L15_1'] = dat['loc_R_L15_1'] -36
    dat['loc_R_L15_2'] = dat['loc_R_L15_2'] -36
    dat['loc_R_R15_1'] = dat['loc_R_R15_1'] -36
    dat['loc_R_R15_2'] = dat['loc_R_R15_2'] -36
    dat['loc_R_R24_1'] = dat['loc_R_R24_1'] -36
    dat['loc_R_R24_2'] = dat['loc_R_R24_2'] -36
    
    
    sns.swarmplot(data=dat)
    ax2.set_xlabel('Horizontal Locations')
    ax2.set_ylabel('Perceived Location (cm)')
    x = [0, 1, 2,3,4,5,6,7]
    labels = trials
    plt.xticks(x, labels, rotation='horizontal')
    plt.title('C')
              
                 
    
    ax3 = fig.add_subplot(223)  # Validation_location_L
    dat = df[['loc_L_L24_1', 'loc_L_L24_2', 'loc_L_L15_1', 'loc_L_L15_2', 
              'loc_L_R15_1', 'loc_L_R15_2', 'loc_L_R24_1', 'loc_L_R24_2']].copy()
    dat['loc_L_L24_1'] = dat['loc_L_L24_1'] -36
    dat['loc_L_L24_1'] = dat['loc_L_L24_1'] *-1
    dat['loc_L_L24_2'] = dat['loc_L_L24_2'] -36
    dat['loc_L_L24_2'] = dat['loc_L_L24_2'] *-1
    dat['loc_L_L15_1'] = dat['loc_L_L15_1'] -36
    dat['loc_L_L15_1'] = dat['loc_L_L15_1'] *-1
    dat['loc_L_L15_2'] = dat['loc_L_L15_2'] -36
    dat['loc_L_L15_2'] = dat['loc_L_L15_2'] *-1
    dat['loc_L_R15_1'] = dat['loc_L_R15_1'] -36
    dat['loc_L_R15_2'] = dat['loc_L_R15_2'] -36
    dat['loc_L_R24_1'] = dat['loc_L_R24_1'] -36
    dat['loc_L_R24_2'] = dat['loc_L_R24_2'] -36
    
    
    sns.swarmplot(data=dat)
    ax3.set_xlabel('Horizontal Locations')
    ax3.set_ylabel('Perceived Location (cm)')
    x = [0, 1, 2,3,4,5,6,7]
    labels = trials
    plt.xticks(x, labels, rotation='horizontal')
    plt.title('B')
                  
#---------------------------------------
# ACTUAL OUTCOMES
# --------------------------------------





# Figure 1A
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
    
    
    
    
    

# FIGURE 2A
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
#    x_diffr = x_diff+1
#    a1 = df['-15_vs_-24_sp']
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

    x_val = x_g + max(jit_b) + 0.1
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

    x_val = x_g + max(jit_b) + 0.1
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

    x_val = x_g + max(jit_b) + 0.1
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

    x_val = x_g + max(jit_b) + 0.1
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
