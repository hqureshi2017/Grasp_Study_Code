# LOCATION OF L and R for the VALIDATION DATA (see below for NG)

# This could be used to further inform us for the HinB study

# Code also includes difference between validation and NO GRASP data. 

# Code also includes difference between 15 cm and 24 - is proprioceptive drift
# different the closer/further you are from midline

# NOTE: For 15 and 24, Left hand location has been *-1. 
#                      Right hand has NOT!!

import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from scipy.stats import t
from numpy import max
from numpy import mean
from numpy import std
from jitter import jitter
import numpy as np 


  # Plotting the L and R for each position
  
t_val = float(t.ppf([0.975], 30))

fig = plt.figure(figsize=(15, 9))
ax1 = fig.add_subplot(111)

# PLOTTING SPACING -24cm
x_l = 1
x_r = 1.5


a = [df['loc_L_R24_1'].values] + [df['loc_L_R24_2'].values] 
b = [df['loc_R_R24_1'].values] + [df['loc_R_R24_2'].values] 
a = [item for sublist in a for item in sublist]
b = [item for sublist in b for item in sublist]
dat = []
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)

# LEFT HAND
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_l], y1, 'ok', color='k', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_l-.05], mean_a, 'ok', markersize=10, marker='^')
plt.plot([x_l-.05, x_l-.05], mean_a_CI, '-k', linewidth=2)

# RIGHT HAND

for x1, y1, in zip(jit_b, b):
    plt.plot([x1+x_r], y1, 'ok', color='k', marker='^')

mean_b = mean(b)
sd_b = std(b)
n = 30
mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
             mean_b + (sd_b / sqrt(n)) * t_val]
plt.plot([x_r -.05], mean_b, 'ok', markersize=10, marker='^')
plt.plot([x_r -.05, x_r-.05], mean_b_CI, '-k', linewidth=2)


# PLOTTING SPACING -15cm
x_l = x_l+1
x_r = x_r+1


a = [df['loc_L_R15_1'].values] + [df['loc_L_R15_2'].values] 
b = [df['loc_R_R15_1'].values] + [df['loc_R_R15_2'].values] 
a = [item for sublist in a for item in sublist]
b = [item for sublist in b for item in sublist]
dat = []
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)

# LEFT HAND
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_l], y1, 'ok', color='k', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_l-.05], mean_a, 'ok', markersize=10, marker='^')
plt.plot([x_l-.05, x_l-.05], mean_a_CI, '-k', linewidth=2)

# RIGHT HAND

for x1, y1, in zip(jit_b, b):
    plt.plot([x1+x_r], y1, 'ok', color='k', marker='^')

mean_b = mean(b)
sd_b = std(b)
n = 30
mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
             mean_b + (sd_b / sqrt(n)) * t_val]
plt.plot([x_r -.05], mean_b, 'ok', markersize=10, marker='^')
plt.plot([x_r -.05, x_r-.05], mean_b_CI, '-k', linewidth=2)


# PLOTTING SPACING 15cm
x_l = x_l+1
x_r = x_r+1


a = [df['loc_L_L15_1'].values] + [df['loc_L_L15_2'].values] 
b = [df['loc_R_L15_1'].values] + [df['loc_R_L15_2'].values] 
a = [item*-1 for sublist in a for item in sublist]
b = [item for sublist in b for item in sublist]
dat = []
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)

# LEFT HAND
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_l], y1, 'ok', color='k', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_l-.05], mean_a, 'ok', markersize=10, marker='^')
plt.plot([x_l-.05, x_l-.05], mean_a_CI, '-k', linewidth=2)

# RIGHT HAND

for x1, y1, in zip(jit_b, b):
    plt.plot([x1+x_r], y1, 'ok', color='k', marker='^')

mean_b = mean(b)
sd_b = std(b)
n = 30
mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
             mean_b + (sd_b / sqrt(n)) * t_val]
plt.plot([x_r -.05], mean_b, 'ok', markersize=10, marker='^')
plt.plot([x_r -.05, x_r-.05], mean_b_CI, '-k', linewidth=2)

# PLOTTING SPACING 24cm
x_l = x_l+1
x_r = x_r+1


a = [df['loc_L_L24_1'].values] + [df['loc_L_L24_2'].values] 
b = [df['loc_R_L24_1'].values] + [df['loc_R_L24_2'].values] 
a = [item*-1 for sublist in a for item in sublist]
b = [item for sublist in b for item in sublist]
dat = []
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)

# LEFT HAND
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_l], y1, 'ok', color='k', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_l-.05], mean_a, 'ok', markersize=10, marker='^')
plt.plot([x_l-.05, x_l-.05], mean_a_CI, '-k', linewidth=2)

# RIGHT HAND

for x1, y1, in zip(jit_b, b):
    plt.plot([x1+x_r], y1, 'ok', color='k', marker='^')

mean_b = mean(b)
sd_b = std(b)
n = 30
mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
             mean_b + (sd_b / sqrt(n)) * t_val]
plt.plot([x_r -.05], mean_b, 'ok', markersize=10, marker='^')
plt.plot([x_r -.05, x_r-.05], mean_b_CI, '-k', linewidth=2)


plt.plot([.5,5], [24,24], '--k')    
plt.plot([.5,5], [15,15], '--k')    
plt.plot([.5,5], [0,0], '--k')    

xticks = [1 ,1.25, 1.5, 2,2.25, 2.5, 3, 3.25, 3.5, 4, 4.25, 4.5]
labels = ['L','-24', 'R', 'L', '-15', 'R','L', '15', 'R','L', '24', 'R']
plt.xticks(xticks, labels)
plt.ylim([-10,32])

va = [ 0, -.05, 0, 0, -.05, 0,
       0, -.05, 0, 0, -.05, 0]
for z, y in zip( ax1.get_xticklabels( ), va ):
    z.set_y( y )

    
    #Removing ticks from axes
ax1.tick_params( axis='x', which='major', bottom='off', top='off', labelbottom='on', left='off', right='off', labelsize=14)
ax1.tick_params( axis='y', which='major', direction='out', length=3, bottom='off', top='off', labelbottom='on', right='off', labelsize=14)
    
    
plt.title('')
ax1.set_xlabel('')
ax1.set_ylabel('Perceived location (cm)', size=20)
plt.savefig('./Graphs/data/validation_loc.pdf')
plt.savefig('./Graphs/data/validation_loc.svg')


#-------------------------------------
#  FIGURE 1B - Difference from reality
#-------------------------------------


  # Plotting the L and R vs reality for each position
  
t_val = float(t.ppf([0.975], 30))

fig = plt.figure(figsize=(15, 9))
ax1 = fig.add_subplot(111)

# PLOTTING SPACING -24cm
x_l = 1
x_r = 1.5


a = [df['loc_L_R24_1'].values-24] + [df['loc_L_R24_2'].values-24] 
b = [df['loc_R_R24_1'].values] + [df['loc_R_R24_2'].values] 
a = [item for sublist in a for item in sublist]
b = [item for sublist in b for item in sublist]
dat = []
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)

# LEFT HAND
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_l], y1, 'ok', color='k', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_l-.05], mean_a, 'ok', markersize=10, marker='^')
plt.plot([x_l-.05, x_l-.05], mean_a_CI, '-k', linewidth=2)

# RIGHT HAND

for x1, y1, in zip(jit_b, b):
    plt.plot([x1+x_r], y1, 'ok', color='k', marker='^')

mean_b = mean(b)
sd_b = std(b)
n = 30
mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
             mean_b + (sd_b / sqrt(n)) * t_val]
plt.plot([x_r -.05], mean_b, 'ok', markersize=10, marker='^')
plt.plot([x_r -.05, x_r-.05], mean_b_CI, '-k', linewidth=2)


# PLOTTING SPACING -15cm
x_l = x_l+1
x_r = x_r+1


a = [df['loc_L_R15_1'].values-15] + [df['loc_L_R15_2'].values-15] 
b = [df['loc_R_R15_1'].values] + [df['loc_R_R15_2'].values] 
a = [item for sublist in a for item in sublist]
b = [item for sublist in b for item in sublist]
dat = []
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)

# LEFT HAND
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_l], y1, 'ok', color='k', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_l-.05], mean_a, 'ok', markersize=10, marker='^')
plt.plot([x_l-.05, x_l-.05], mean_a_CI, '-k', linewidth=2)

# RIGHT HAND

for x1, y1, in zip(jit_b, b):
    plt.plot([x1+x_r], y1, 'ok', color='k', marker='^')

mean_b = mean(b)
sd_b = std(b)
n = 30
mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
             mean_b + (sd_b / sqrt(n)) * t_val]
plt.plot([x_r -.05], mean_b, 'ok', markersize=10, marker='^')
plt.plot([x_r -.05, x_r-.05], mean_b_CI, '-k', linewidth=2)


# PLOTTING SPACING 15cm
x_l = x_l+1
x_r = x_r+1


a = [df['loc_L_L15_1'].values--15] + [df['loc_L_L15_2'].values--15] 
b = [df['loc_R_L15_1'].values] + [df['loc_R_L15_2'].values] 
a = [item*-1 for sublist in a for item in sublist]
b = [item for sublist in b for item in sublist]
dat = []
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)

# LEFT HAND
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_l], y1, 'ok', color='k', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_l-.05], mean_a, 'ok', markersize=10, marker='^')
plt.plot([x_l-.05, x_l-.05], mean_a_CI, '-k', linewidth=2)

# RIGHT HAND

for x1, y1, in zip(jit_b, b):
    plt.plot([x1+x_r], y1, 'ok', color='k', marker='^')

mean_b = mean(b)
sd_b = std(b)
n = 30
mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
             mean_b + (sd_b / sqrt(n)) * t_val]
plt.plot([x_r -.05], mean_b, 'ok', markersize=10, marker='^')
plt.plot([x_r -.05, x_r-.05], mean_b_CI, '-k', linewidth=2)

# PLOTTING SPACING 24cm
x_l = x_l+1
x_r = x_r+1


a = [df['loc_L_L24_1'].values--24] + [df['loc_L_L24_2'].values--24] 
b = [df['loc_R_L24_1'].values] + [df['loc_R_L24_2'].values] 
a = [item*-1 for sublist in a for item in sublist]
b = [item for sublist in b for item in sublist]
dat = []
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)

# LEFT HAND
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_l], y1, 'ok', color='k', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_l-.05], mean_a, 'ok', markersize=10, marker='^')
plt.plot([x_l-.05, x_l-.05], mean_a_CI, '-k', linewidth=2)

# RIGHT HAND

for x1, y1, in zip(jit_b, b):
    plt.plot([x1+x_r], y1, 'ok', color='k', marker='^')

mean_b = mean(b)
sd_b = std(b)
n = 30
mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
             mean_b + (sd_b / sqrt(n)) * t_val]
plt.plot([x_r -.05], mean_b, 'ok', markersize=10, marker='^')
plt.plot([x_r -.05, x_r-.05], mean_b_CI, '-k', linewidth=2)

   
plt.plot([.5,5], [0,0], '--k')    

xticks = [1 ,1.25, 1.5, 2,2.25, 2.5, 3, 3.25, 3.5, 4, 4.25, 4.5]
labels = ['L','-24', 'R', 'L', '-15', 'R','L', '15', 'R','L', '24', 'R']
plt.xticks(xticks, labels)

va = [ 0, -.05, 0, 0, -.05, 0,
       0, -.05, 0, 0, -.05, 0]
for z, y in zip( ax1.get_xticklabels( ), va ):
    z.set_y( y )

#Removing ticks from axes
ax1.tick_params( axis='x', which='major', bottom='off', top='off', labelbottom='on', left='off', right='off', labelsize=14)
ax1.tick_params( axis='y', which='major', direction='out', length=3, bottom='off', top='off', labelbottom='on', right='off', labelsize=14)
    
    
    
plt.title('')
ax1.set_xlabel('')
ax1.set_ylabel('Difference in perceived location (cm)', size=20)
plt.savefig('./Graphs/data/fig1B_loc_vs_reality.pdf')
plt.savefig('./Graphs/data/fig1B_loc_vs_reality.svg')


#---------------------
#   PLOTTING FOR NG
#---------------------



  # Plotting the L and R for each position
  
t_val = float(t.ppf([0.975], 30))

fig = plt.figure(figsize=(15, 9))
ax1 = fig.add_subplot(111)

# PLOTTING SPACING -24cm
x_l = 1
x_r = 1.5


a = [df['nograsp_c24cm_l'].values] 
b = [df['nograsp_c24cm_r'].values]  
a = [item for sublist in a for item in sublist]
b = [item for sublist in b for item in sublist]
dat = []
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)

# LEFT HAND
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_l], y1, 'ok', color='k', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_l-.05], mean_a, 'ok', markersize=10, marker='^')
plt.plot([x_l-.05, x_l-.05], mean_a_CI, '-k', linewidth=2)

# RIGHT HAND

for x1, y1, in zip(jit_b, b):
    plt.plot([x1+x_r], y1, 'ok', color='k', marker='^')

mean_b = mean(b)
sd_b = std(b)
n = 30
mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
             mean_b + (sd_b / sqrt(n)) * t_val]
plt.plot([x_r -.05], mean_b, 'ok', markersize=10, marker='^')
plt.plot([x_r -.05, x_r-.05], mean_b_CI, '-k', linewidth=2)


# PLOTTING SPACING -15cm
x_l = x_l+1
x_r = x_r+1


a = [df['nograsp_c15cm_l'].values] 
b = [df['nograsp_c15cm_r'].values]  
a = [item for sublist in a for item in sublist]
b = [item for sublist in b for item in sublist]
dat = []
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)

# LEFT HAND
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_l], y1, 'ok', color='k', marker='^')

mean_a = np.nanmean(a)
sd_a = np.nanstd(a)
n = 29
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_l-.05], mean_a, 'ok', markersize=10, marker='^')
plt.plot([x_l-.05, x_l-.05], mean_a_CI, '-k', linewidth=2)

# RIGHT HAND

for x1, y1, in zip(jit_b, b):
    plt.plot([x1+x_r], y1, 'ok', color='k', marker='^')

mean_b = np.nanmean(b)
sd_b = np.nanstd(b)
n = 29
mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
             mean_b + (sd_b / sqrt(n)) * t_val]
plt.plot([x_r -.05], mean_b, 'ok', markersize=10, marker='^')
plt.plot([x_r -.05, x_r-.05], mean_b_CI, '-k', linewidth=2)


# PLOTTING SPACING 15cm
x_l = x_l+1
x_r = x_r+1


a = [df['nograsp_uc15cm_l'].values] 
b = [df['nograsp_uc15cm_r'].values]  
a = [item*-1 for sublist in a for item in sublist]
b = [item for sublist in b for item in sublist]
dat = []
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)

# LEFT HAND
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_l], y1, 'ok', color='k', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_l-.05], mean_a, 'ok', markersize=10, marker='^')
plt.plot([x_l-.05, x_l-.05], mean_a_CI, '-k', linewidth=2)

# RIGHT HAND

for x1, y1, in zip(jit_b, b):
    plt.plot([x1+x_r], y1, 'ok', color='k', marker='^')

mean_b = mean(b)
sd_b = std(b)
n = 30
mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
             mean_b + (sd_b / sqrt(n)) * t_val]
plt.plot([x_r -.05], mean_b, 'ok', markersize=10, marker='^')
plt.plot([x_r -.05, x_r-.05], mean_b_CI, '-k', linewidth=2)

# PLOTTING SPACING 24cm
x_l = x_l+1
x_r = x_r+1


a = [df['nograsp_uc24cm_l'].values] 
b = [df['nograsp_uc24cm_r'].values]  
a = [item*-1 for sublist in a for item in sublist]
b = [item for sublist in b for item in sublist]
dat = []
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)

# LEFT HAND
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_l], y1, 'ok', color='k', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_l-.05], mean_a, 'ok', markersize=10, marker='^')
plt.plot([x_l-.05, x_l-.05], mean_a_CI, '-k', linewidth=2)

# RIGHT HAND

for x1, y1, in zip(jit_b, b):
    plt.plot([x1+x_r], y1, 'ok', color='k', marker='^')

mean_b = mean(b)
sd_b = std(b)
n = 30
mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
             mean_b + (sd_b / sqrt(n)) * t_val]
plt.plot([x_r -.05], mean_b, 'ok', markersize=10, marker='^')
plt.plot([x_r -.05, x_r-.05], mean_b_CI, '-k', linewidth=2)


plt.plot([.5,5], [24,24], '--k')    
plt.plot([.5,5], [15,15], '--k')    
plt.plot([.5,5], [0,0], '--k')    

xticks = [1 ,1.25, 1.5, 2,2.25, 2.5, 3, 3.25, 3.5, 4, 4.25, 4.5]
labels = ['L','-24', 'R', 'L', '-15', 'R','L', '15', 'R','L', '24', 'R']
plt.xticks(xticks, labels)

va = [ 0, -.05, 0, 0, -.05, 0,
       0, -.05, 0, 0, -.05, 0]
for z, y in zip( ax1.get_xticklabels( ), va ):
    z.set_y( y )
    
#Removing ticks from axes
ax1.tick_params( axis='x', which='major', bottom='off', top='off', labelbottom='on', left='off', right='off', labelsize=14)
ax1.tick_params( axis='y', which='major', direction='out', length=3, bottom='off', top='off', labelbottom='on', right='off', labelsize=14)
    
plt.title('')
ax1.set_xlabel('')
ax1.set_ylabel('Perceived location (cm)', size=20)
plt.savefig('./Graphs/data/fig4_NG_loc.pdf')
plt.savefig('./Graphs/data/fig4_NG_loc.svg')




#-------------------------------------
#  Difference from reality FOR NG
#-------------------------------------


  # Plotting the L and R vs reality for each position in NG
  
  

t_val = float(t.ppf([0.975], 30))

fig = plt.figure(figsize=(15, 9))
ax1 = fig.add_subplot(111)

# PLOTTING SPACING -24cm
x_l = 1
x_r = 1.5


a = [df['nograsp_c24cm_l'].values-24] 
b = [df['nograsp_c24cm_r'].values]  
a = [item for sublist in a for item in sublist]
b = [item for sublist in b for item in sublist]
dat = []
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)

# LEFT HAND
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_l], y1, 'ok', color='k', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_l-.05], mean_a, 'ok', markersize=10, marker='^')
plt.plot([x_l-.05, x_l-.05], mean_a_CI, '-k', linewidth=2)

# RIGHT HAND

for x1, y1, in zip(jit_b, b):
    plt.plot([x1+x_r], y1, 'ok', color='k', marker='^')

mean_b = mean(b)
sd_b = std(b)
n = 30
mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
             mean_b + (sd_b / sqrt(n)) * t_val]
plt.plot([x_r -.05], mean_b, 'ok', markersize=10, marker='^')
plt.plot([x_r -.05, x_r-.05], mean_b_CI, '-k', linewidth=2)


# PLOTTING SPACING -15cm
x_l = x_l+1
x_r = x_r+1


a = [df['nograsp_c15cm_l'].values-15] 
b = [df['nograsp_c15cm_r'].values]  
a = [item for sublist in a for item in sublist]
b = [item for sublist in b for item in sublist]
dat = []
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)

# LEFT HAND
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_l], y1, 'ok', color='k', marker='^')

mean_a = np.nanmean(a)
sd_a = np.nanstd(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_l-.05], mean_a, 'ok', markersize=10, marker='^')
plt.plot([x_l-.05, x_l-.05], mean_a_CI, '-k', linewidth=2)

# RIGHT HAND

for x1, y1, in zip(jit_b, b):
    plt.plot([x1+x_r], y1, 'ok', color='k', marker='^')

mean_b = np.nanmean(b)
sd_b = np.nanstd(b)
n = 30
mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
             mean_b + (sd_b / sqrt(n)) * t_val]
plt.plot([x_r -.05], mean_b, 'ok', markersize=10, marker='^')
plt.plot([x_r -.05, x_r-.05], mean_b_CI, '-k', linewidth=2)


# PLOTTING SPACING 15cm
x_l = x_l+1
x_r = x_r+1


a = [df['nograsp_uc15cm_l'].values--15] 
b = [df['nograsp_uc15cm_r'].values]  
a = [item*-1 for sublist in a for item in sublist]
b = [item for sublist in b for item in sublist]
dat = []
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)

# LEFT HAND
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_l], y1, 'ok', color='k', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_l-.05], mean_a, 'ok', markersize=10, marker='^')
plt.plot([x_l-.05, x_l-.05], mean_a_CI, '-k', linewidth=2)

# RIGHT HAND

for x1, y1, in zip(jit_b, b):
    plt.plot([x1+x_r], y1, 'ok', color='k', marker='^')

mean_b = mean(b)
sd_b = std(b)
n = 30
mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
             mean_b + (sd_b / sqrt(n)) * t_val]
plt.plot([x_r -.05], mean_b, 'ok', markersize=10, marker='^')
plt.plot([x_r -.05, x_r-.05], mean_b_CI, '-k', linewidth=2)

# PLOTTING SPACING 24cm
x_l = x_l+1
x_r = x_r+1


a = [df['nograsp_uc24cm_l'].values--24] 
b = [df['nograsp_uc24cm_r'].values]  
a = [item*-1 for sublist in a for item in sublist]
b = [item for sublist in b for item in sublist]
dat = []
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)

# LEFT HAND
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_l], y1, 'ok', color='k', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_l-.05], mean_a, 'ok', markersize=10, marker='^')
plt.plot([x_l-.05, x_l-.05], mean_a_CI, '-k', linewidth=2)

# RIGHT HAND

for x1, y1, in zip(jit_b, b):
    plt.plot([x1+x_r], y1, 'ok', color='k', marker='^')

mean_b = mean(b)
sd_b = std(b)
n = 30
mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
             mean_b + (sd_b / sqrt(n)) * t_val]
plt.plot([x_r -.05], mean_b, 'ok', markersize=10, marker='^')
plt.plot([x_r -.05, x_r-.05], mean_b_CI, '-k', linewidth=2)

   
plt.plot([.5,5], [0,0], '--k')    

xticks = [1 ,1.25, 1.5, 2,2.25, 2.5, 3, 3.25, 3.5, 4, 4.25, 4.5]
labels = ['L','-24', 'R', 'L', '-15', 'R','L', '15', 'R','L', '24', 'R']
plt.xticks(xticks, labels)

va = [ 0, -.05, 0, 0, -.05, 0,
       0, -.05, 0, 0, -.05, 0]
for z, y in zip( ax1.get_xticklabels( ), va ):
    z.set_y( y )

#Removing ticks from axes
ax1.tick_params( axis='x', which='major', bottom='off', top='off', labelbottom='on', left='off', right='off', labelsize=14)
ax1.tick_params( axis='y', which='major', direction='out', length=3, bottom='off', top='off', labelbottom='on', right='off', labelsize=14)
    
    
plt.title('')
ax1.set_xlabel('')
ax1.set_ylabel('Perceived Location (cm)',size=20)
plt.savefig('./Graphs/data/loc_vs_reality_NG.pdf')
plt.savefig('./Graphs/data/loc_vs_reality_NG.svg')












#-------------------------------------
#  FIGURE - Difference between validation and NG
#-------------------------------------
# Comparing the difference between validation and NO GRASP condition for the perceived
# location of left hand. 

fig = plt.figure(figsize=(15, 9))
ax1 = fig.add_subplot(111)

t_val = float(t.ppf([0.975], 30))

# PLOTTING difference -24cm
x_diff = 1

# location of left in validation
a = [df['loc_L_R24_1'].values] + [df['loc_L_R24_2'].values] 
a = [item for sublist in a for item in sublist]
# location of left in NG
b = [df['nograsp_c24cm_l'].values]*2
b = [item for sublist in b for item in sublist]


# difference between the location of left hand for validation and NG
a =  [x1 - x2 for (x1, x2) in zip(a, b)]
b =  [x1 - x2 for (x1, x2) in zip(a, b)]
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_diff], y1, 'ok', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_diff-.05], mean_a, 'ok', markersize=15, marker='^')
plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)

   
# PLOTTING difference -15cm
x_diff = 2


# location of left in validation
a = [df['loc_L_R15_1'].values] + [df['loc_L_R15_2'].values] 
a = [item for sublist in a for item in sublist]
# location of left in NG
b = [df['nograsp_c15cm_l'].values]*2
b = [item for sublist in b for item in sublist]


# difference between the location of left hand for validation and NG
a =  [x1 - x2 for (x1, x2) in zip(a, b)]
b =  [x1 - x2 for (x1, x2) in zip(a, b)]
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_diff], y1, 'ok', marker='^')

mean_a = np.nanmean(a)
sd_a = np.nanstd(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_diff-.05], mean_a, 'ok', markersize=15, marker='^')
plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)


# PLOTTING difference 15cm
x_diff = 3

# location of left in validation
a = [df['loc_L_L15_1'].values] + [df['loc_L_L15_2'].values] 
a = [item for sublist in a for item in sublist]
# location of left in NG
b = [df['nograsp_uc15cm_l'].values]*2
b = [item for sublist in b for item in sublist]


# difference between the location of left hand for validation and NG
a =  [x1 - x2 for (x1, x2) in zip(a, b)]
b =  [x1 - x2 for (x1, x2) in zip(a, b)]
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_diff], y1, 'ok', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_diff-.05], mean_a, 'ok', markersize=15, marker='^')
plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)


# PLOTTING difference 24cm
x_diff = 4

# location of left in validation
a = [df['loc_L_L24_1'].values] + [df['loc_L_L24_2'].values] 
a = [item for sublist in a for item in sublist]
# location of left in NG
b = [df['nograsp_uc24cm_l'].values]*2
b = [item for sublist in b for item in sublist]


# difference between the location of left hand for validation and NG
a =  [x1 - x2 for (x1, x2) in zip(a, b)]
b =  [x1 - x2 for (x1, x2) in zip(a, b)]
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_diff], y1, 'ok', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_diff-.05], mean_a, 'ok', markersize=15, marker='^')
plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)

plt.plot([4.2,.8], [0,0], '--k', alpha=.5)    

# Hide the right and top spines
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
    
#Removing ticks from axes
ax1.tick_params( axis='x', which='major', bottom='off', top='off', labelbottom='on', left='off', right='off', labelsize=14)
ax1.tick_params( axis='y', which='major', direction='out', length=3, bottom='off', top='off', labelbottom='on', right='off', labelsize=14)

xticks = [1, 1.5, 2, 3, 3.5, 4 ]
labels = ['-24 cm', 'Crossed', '-15 cm',
          '15 cm', 'Uncrossed','24 cm']
plt.xticks(xticks, labels)

# vertical alignment of xtick labels
va = [ 0, -.05, 0, 0, -.05, 0, 
      0, -.05, 0, 0, -.05, 0, ]
for z, y in zip( ax1.get_xticklabels( ), va ):
    z.set_y( y )
    
ax1.set_xlabel('')
ax1.set_ylabel('Difference in location (cm)',size=20)

plt.title('')
plt.savefig('./Graphs/data/' + 'fig_2C.pdf')
plt.savefig('./Graphs/data/' + 'fig_2C.svg')









#-------------------------------------
#  FIGURE - Difference between 15 cm and 24 
#-------------------------------------
# Comparing the difference in proprioceptive drift (i.e perceived location of L hand vs reality) between 15 cm and 24 cm.
# We want to know whether drift is different across locations and hence whether we need both for the HandinBox study or is one enough?


fig = plt.figure(figsize=(15, 9))
ax1 = fig.add_subplot(111)

t_val = float(t.ppf([0.975], 30))

# PLOTTING difference -24cm vs -15cm (loc of L hand vs reality)
x_diff = 1

# location of left in -24
a = [df['loc_L_R24_1'].values-24] + [df['loc_L_R24_2'].values-24] 
a = [item for sublist in a for item in sublist]
# location of left in -15
b = [df['loc_L_R15_1'].values-15] + [df['loc_L_R15_2'].values-15] 
b = [item for sublist in b for item in sublist]

# difference between the location of left hand for -24 and -15
a =  [x1 - x2 for (x1, x2) in zip(a, b)]
b =  [x1 - x2 for (x1, x2) in zip(a, b)]
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_diff], y1, 'ok', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_diff-.05], mean_a, 'ok', markersize=15, marker='^')
plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)


# PLOTTING difference 15cm vs 24cm (loc of L hand vs reality)
x_diff = 1.5

# location of left in 15
a = [df['loc_L_L15_1'].values--15] + [df['loc_L_L15_2'].values--15] 
a = [item*-1 for sublist in a for item in sublist]
# location of left in 24
b = [df['loc_L_L24_1'].values--24] + [df['loc_L_L24_2'].values--24] 
b = [item*-1 for sublist in b for item in sublist]

# difference between the location of left hand for 24 and 15
a =  [x1 - x2 for (x1, x2) in zip(a, b)]
b =  [x1 - x2 for (x1, x2) in zip(a, b)]
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_diff], y1, 'ok', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_diff-.05], mean_a, 'ok', markersize=15, marker='^')
plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)


plt.plot([1.7,.8], [0,0], '--k', alpha=.5)    

# Hide the right and top spines
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
    
#Removing ticks from axes
ax1.tick_params( axis='x', which='major', bottom='off', top='off', labelbottom='on', left='off', right='off', labelsize=14)
ax1.tick_params( axis='y', which='major', direction='out', length=3, bottom='off', top='off', labelbottom='on', right='off', labelsize=14)

xticks = [1, 1.5]
labels = ['-24cm vs -15cm', '15cm vs 24cm']
plt.xticks(xticks, labels)
plt.ylim(-7,10)
    
ax1.set_xlabel('')
ax1.set_ylabel('Effect of distance on drift (cm)',size=20)

plt.title('')
plt.savefig('./Graphs/data/' + 'fig_2C.pdf')
plt.savefig('./Graphs/data/' + 'fig_2C.svg')
