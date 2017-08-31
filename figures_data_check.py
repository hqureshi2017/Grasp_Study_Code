
########### HASSAN ##############
#   File contains code for making data visualisation graphs. 
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
                  