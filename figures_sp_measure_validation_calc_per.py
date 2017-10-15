# Calc and Per Spacings for VALIDATION DATA (see below for NG data also)

# Code also includes difference between val and NG

# This could be used to further inform us for the HinB study


import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from scipy.stats import t
from numpy import max
from numpy import mean
from numpy import std
from jitter import jitter
import numpy as np
import grasp_study_code as g
df = g.create()

  # Plotting the Calc and Per for each position
  
t_val = float(t.ppf([0.975], 30))

fig = plt.figure(figsize=(15, 9))
ax1 = fig.add_subplot(111)

# PLOTTING SPACING -24cm
x_l = 1
x_r = 1.5


a = [df['sp_calc_c24_1'].values] + [df['sp_calc_c24_2'].values] 
b = [df['R24_1'].values] + [df['R24_2'].values] 
a = [item for sublist in a for item in sublist]
b = [item for sublist in b for item in sublist]
dat = []
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)

# Calculated
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_l], y1, 'ok', color='k', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_l-.05], mean_a, 'ok', markersize=10, marker='^')
plt.plot([x_l-.05, x_l-.05], mean_a_CI, '-k', linewidth=2)

# Perceived

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


a = [df['sp_calc_c15_1'].values] + [df['sp_calc_c15_2'].values] 
b = [df['R15_1'].values] + [df['R15_2'].values] 
a = [item for sublist in a for item in sublist]
b = [item for sublist in b for item in sublist]
dat = []
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)

# Calculated
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_l], y1, 'ok', color='k', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_l-.05], mean_a, 'ok', markersize=10, marker='^')
plt.plot([x_l-.05, x_l-.05], mean_a_CI, '-k', linewidth=2)

# Perceived


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


a = [df['sp_calc_uc15_1'].values] + [df['sp_calc_uc15_2'].values] 
b = [df['L15_1'].values] + [df['L15_2'].values] 
a = [item for sublist in a for item in sublist]
b = [item for sublist in b for item in sublist]
dat = []
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)

# Calculated
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_l], y1, 'ok', color='k', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_l-.05], mean_a, 'ok', markersize=10, marker='^')
plt.plot([x_l-.05, x_l-.05], mean_a_CI, '-k', linewidth=2)

# Perceived


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


a = [df['sp_calc_uc24_1'].values] + [df['sp_calc_uc24_2'].values] 
b = [df['L24_1'].values] + [df['L24_2'].values] 
a = [item for sublist in a for item in sublist]
b = [item for sublist in b for item in sublist]
dat = []
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)

# Calculated
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_l], y1, 'ok', color='k', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_l-.05], mean_a, 'ok', markersize=10, marker='^')
plt.plot([x_l-.05, x_l-.05], mean_a_CI, '-k', linewidth=2)

# Perceived

for x1, y1, in zip(jit_b, b):
    plt.plot([x1+x_r], y1, 'ok', color='k', marker='^')

mean_b = mean(b)
sd_b = std(b)
n = 30
mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
             mean_b + (sd_b / sqrt(n)) * t_val]
plt.plot([x_r -.05], mean_b, 'ok', markersize=10, marker='^')
plt.plot([x_r -.05, x_r-.05], mean_b_CI, '-k', linewidth=2)

#Removing ticks from axes
ax1.tick_params( axis='x', which='major', bottom='off', top='off', labelbottom='on', left='off', right='off', labelsize=14)
ax1.tick_params( axis='y', which='major', direction='out', length=3, bottom='off', top='off', labelbottom='on', right='off', labelsize=14)

xticks = [1 ,1.25, 1.5, 2,2.25, 2.5, 3, 3.25, 3.5, 4, 4.25, 4.5]
labels = ['Cal','-24', 'Per', 'Cal', '-15', 'Per','Cal', '15', 'Per','Cal', '24', 'Per']
plt.xticks(xticks, labels)

va = [ 0, -.05, 0, 0, -.05, 0,
       0, -.05, 0, 0, -.05, 0]
for z, y in zip( ax1.get_xticklabels( ), va ):
    z.set_y( y )
    
plt.plot([.5,5], [24,24], '--k',alpha=.5)    
plt.plot([.5,5], [15,15], '--k', alpha=.5)  

    
plt.title('')
ax1.set_xlabel('')
ax1.set_ylabel('Spacing (cm)', size=20)
plt.savefig('./Graphs/data/validation_measures.pdf')
plt.savefig('./Graphs/data/validation_measures.svg')


#---------------------------
#    Calc and Per vs Reality
#---------------------------

  # Plotting the Calc and Per for each position
  
t_val = float(t.ppf([0.975], 30))

fig = plt.figure(figsize=(15, 9))
ax1 = fig.add_subplot(111)

# PLOTTING SPACING -24cm
x_l = 1
x_r = 1.5


a = [df['sp_calc_c24_1'].values-24] + [df['sp_calc_c24_2'].values-24] 
b = [df['R24_1'].values-24] + [df['R24_2'].values-24] 
a = [item for sublist in a for item in sublist]
b = [item for sublist in b for item in sublist]
dat = []
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)

# Calculated
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_l], y1, 'ok', color='r', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_l-.05], mean_a, 'ok', markersize=10, marker='^')
plt.plot([x_l-.05, x_l-.05], mean_a_CI, '-k', linewidth=2)

# Perceived

for x1, y1, in zip(jit_b, b):
    plt.plot([x1+x_r], y1, 'ok', color='r', marker='^')

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


a = [df['sp_calc_c15_1'].values-15] + [df['sp_calc_c15_2'].values-15] 
b = [df['R15_1'].values-15] + [df['R15_2'].values-15] 
a = [item for sublist in a for item in sublist]
b = [item for sublist in b for item in sublist]
dat = []
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)

# Calculated
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_l], y1, 'ok', color='r', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_l-.05], mean_a, 'ok', markersize=10, marker='^')
plt.plot([x_l-.05, x_l-.05], mean_a_CI, '-k', linewidth=2)

# Perceived


for x1, y1, in zip(jit_b, b):
    plt.plot([x1+x_r], y1, 'ok', color='r', marker='^')

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


a = [df['sp_calc_uc15_1'].values-15] + [df['sp_calc_uc15_2'].values-15] 
b = [df['L15_1'].values-15] + [df['L15_2'].values-15] 
a = [item for sublist in a for item in sublist]
b = [item for sublist in b for item in sublist]
dat = []
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)

# Calculated
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_l], y1, 'ok', color='r', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_l-.05], mean_a, 'ok', markersize=10, marker='^')
plt.plot([x_l-.05, x_l-.05], mean_a_CI, '-k', linewidth=2)

# Perceived


for x1, y1, in zip(jit_b, b):
    plt.plot([x1+x_r], y1, 'ok', color='r', marker='^')

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


a = [df['sp_calc_uc24_1'].values-24] + [df['sp_calc_uc24_2'].values-24] 
b = [df['L24_1'].values-24] + [df['L24_2'].values-24] 
a = [item for sublist in a for item in sublist]
b = [item for sublist in b for item in sublist]
dat = []
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)

# Calculated
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_l], y1, 'ok', color='r', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_l-.05], mean_a, 'ok', markersize=10, marker='^')
plt.plot([x_l-.05, x_l-.05], mean_a_CI, '-k', linewidth=2)

# Perceived

for x1, y1, in zip(jit_b, b):
    plt.plot([x1+x_r], y1, 'ok', color='r', marker='^')

mean_b = mean(b)
sd_b = std(b)
n = 30
mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
             mean_b + (sd_b / sqrt(n)) * t_val]
plt.plot([x_r -.05], mean_b, 'ok', markersize=10, marker='^')
plt.plot([x_r -.05, x_r-.05], mean_b_CI, '-k', linewidth=2)

#Removing ticks from axes
ax1.tick_params( axis='x', which='major', bottom='off', top='off', labelbottom='on', left='off', right='off', labelsize=14)
ax1.tick_params( axis='y', which='major', direction='out', length=3, bottom='off', top='off', labelbottom='on', right='off', labelsize=14)

xticks = [1 ,1.25, 1.5, 2,2.25, 2.5, 3, 3.25, 3.5, 4, 4.25, 4.5]
labels = ['Cal','-24', 'Per', 'Cal', '-15', 'Per','Cal', '15', 'Per','Cal', '24', 'Per']
plt.xticks(xticks, labels)

va = [ 0, -.05, 0, 0, -.05, 0,
       0, -.05, 0, 0, -.05, 0]
for z, y in zip( ax1.get_xticklabels( ), va ):
    z.set_y( y )
    


plt.plot([.5,5], [0,0], '-k')   
    
plt.title('')
ax1.set_xlabel('')
ax1.set_ylabel('Spacing (cm)', size=20)
plt.savefig('./Graphs/data/Validation/calc_vs_per_reality_validation.pdf')
plt.savefig('./Graphs/data/Validation/calc_vs_per_reality_validation.svg')

#------------------
#   FOR NG DATA
#------------------


  # Plotting the Calc and Per for each position
  
t_val = float(t.ppf([0.975], 30))

fig = plt.figure(figsize=(15, 9))
ax1 = fig.add_subplot(111)

# PLOTTING SPACING -24cm
x_l = 1
x_r = 1.5


a = [df['-24_calc_spacing_No_Grasp'].values] 
b = [df['nograsp_c24cm_sp'].values]
a = [item for sublist in a for item in sublist]
b = [item for sublist in b for item in sublist]
dat = []
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)

# Calculated
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_l], y1, 'ok', color='k', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_l-.05], mean_a, 'ok', markersize=10, marker='^')
plt.plot([x_l-.05, x_l-.05], mean_a_CI, '-k', linewidth=2)

# Perceived


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


a = [df['-15_calc_spacing_No_Grasp'].values] 
b = [df['nograsp_c15cm_sp'].values] 
a = [item for sublist in a for item in sublist]
b = [item for sublist in b for item in sublist]
dat = []
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)

# Calculated
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_l], y1, 'ok', color='k', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_l-.05], mean_a, 'ok', markersize=10, marker='^')
plt.plot([x_l-.05, x_l-.05], mean_a_CI, '-k', linewidth=2)

# Perceived

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


a = [df['15_calc_spacing_No_Grasp'].values] 
b = [df['nograsp_uc15cm_sp'].values]
a = [item for sublist in a for item in sublist]
b = [item for sublist in b for item in sublist]
dat = []
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)

# Calculated

for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_l], y1, 'ok', color='k', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_l-.05], mean_a, 'ok', markersize=10, marker='^')
plt.plot([x_l-.05, x_l-.05], mean_a_CI, '-k', linewidth=2)

# Perceived

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


a = [df['24_calc_spacing_No_Grasp'].values] 
b = [df['nograsp_uc24cm_sp'].values]
a = [item for sublist in a for item in sublist]
b = [item for sublist in b for item in sublist]
dat = []
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)

# Calculated

for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_l], y1, 'ok', color='k', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_l-.05], mean_a, 'ok', markersize=10, marker='^')
plt.plot([x_l-.05, x_l-.05], mean_a_CI, '-k', linewidth=2)

# Perceived

for x1, y1, in zip(jit_b, b):
    plt.plot([x1+x_r], y1, 'ok', color='k', marker='^')

mean_b = mean(b)
sd_b = std(b)
n = 30
mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
             mean_b + (sd_b / sqrt(n)) * t_val]
plt.plot([x_r -.05], mean_b, 'ok', markersize=10, marker='^')
plt.plot([x_r -.05, x_r-.05], mean_b_CI, '-k', linewidth=2)



xticks = [1 ,1.25, 1.5, 2,2.25, 2.5, 3, 3.25, 3.5, 4, 4.25, 4.5]
labels = ['Cal','-24', 'Per', 'Cal', '-15', 'Per','Cal', '15', 'Per','Cal', '24', 'Per']
plt.xticks(xticks, labels)

va = [ 0, -.05, 0, 0, -.05, 0,
       0, -.05, 0, 0, -.05, 0]
for z, y in zip( ax1.get_xticklabels( ), va ):
    z.set_y( y )
    

plt.plot([.5,5], [24,24], '--k')    
plt.plot([.5,5], [15,15], '--k')   

#Removing ticks from axes
ax1.tick_params( axis='x', which='major', bottom='off', top='off', labelbottom='on', left='off', right='off', labelsize=14)
ax1.tick_params( axis='y', which='major', direction='out', length=3, bottom='off', top='off', labelbottom='on', right='off', labelsize=14)

plt.title('')
ax1.set_xlabel('')
ax1.set_ylabel('Perceived spacing (cm)', size=20)

plt.savefig('./Graphs/data/fig5_NG_measures.pdf')
plt.savefig('./Graphs/data/fig5_NG_measures.svg')







#-------------------------------------
#  FIGURE - Difference between validation and NG
#-------------------------------------
# Comparing the difference between validation and NO GRASP condition for the measures of
# spacing between hands. 

fig = plt.figure(figsize=(15, 9))
ax1 = fig.add_subplot(111)

t_val = float(t.ppf([0.975], 30))

# PLOTTING difference -24cm
x_diff = 1


# Calc spacing between hands in validation
a = [df['sp_calc_c24_1'].values] + [df['sp_calc_c24_2'].values] 
a = [item for sublist in a for item in sublist]
# Calc spacing between hands in NG
b = [df['-24_calc_spacing_No_Grasp'].values]*2
b = [item for sublist in b for item in sublist]


# Per spacing between hands in validation
c = [df['R24_1'].values] + [df['R24_2'].values] 
c = [item for sublist in c for item in sublist]
# Per spacing between hands in NG
d = [df['nograsp_c24cm_sp'].values]*2
d = [item for sublist in d for item in sublist]


# difference between the calc spacing between hands for validation and NG
a =  [x1 - x2 for (x1, x2) in zip(a, b)]
b =  [x1 - x2 for (x1, x2) in zip(c, d)]
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)

# Calc
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_diff], y1, 'ok', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_diff-.05], mean_a, 'ok', markersize=15, marker='^')
plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)

x_diff = x_diff + .2
# Per
for x1, y1, in zip(jit_b, b):
    plt.plot([x1+x_diff], y1, 'ok', marker='^')
    
mean_b = mean(b)
sd_b = std(b)
n = 30
mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
             mean_b + (sd_b / sqrt(n)) * t_val]
plt.plot([x_diff-.05], mean_b, 'ok', markersize=15, marker='^')
plt.plot([x_diff-.05, x_diff-.05], mean_b_CI, '-k', linewidth=2)


   
# PLOTTING difference -15cm
x_diff = 2


# Calc spacing between hands in validation
a = [df['sp_calc_c15_1'].values] + [df['sp_calc_c15_2'].values] 
a = [item for sublist in a for item in sublist]
# Calc spacing between hands in NG
b = [df['-15_calc_spacing_No_Grasp'].values]*2
b = [item for sublist in b for item in sublist]


# Per spacing between hands in validation
c = [df['R15_1'].values] + [df['R15_2'].values] 
c = [item for sublist in c for item in sublist]
# Per spacing between hands in NG
d = [df['nograsp_c15cm_sp'].values]*2
d = [item for sublist in d for item in sublist]


# difference between the calc spacing between hands for validation and NG
a =  [x1 - x2 for (x1, x2) in zip(a, b)]
b =  [x1 - x2 for (x1, x2) in zip(c, d)]
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)

# Calc
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_diff], y1, 'ok', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_diff-.05], mean_a, 'ok', markersize=15, marker='^')
plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)

x_diff = x_diff + .2

# Per
for x1, y1, in zip(jit_b, b):
    plt.plot([x1+x_diff], y1, 'ok', marker='^')

mean_b = mean(b)
sd_b = np.nanstd(b)
n = 60
mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
             mean_b + (sd_b / sqrt(n)) * t_val]
plt.plot([x_diff-.05], mean_b, 'ok', markersize=15, marker='^')
plt.plot([x_diff-.05, x_diff-.05], mean_b_CI, '-k', linewidth=2)




# PLOTTING difference 15cm
x_diff = 3

# Calc spacing between hands in validation
a = [df['sp_calc_uc15_1'].values] + [df['sp_calc_uc15_2'].values] 
a = [item for sublist in a for item in sublist]
# Calc spacing between hands in NG
b = [df['15_calc_spacing_No_Grasp'].values]*2
b = [item for sublist in b for item in sublist]


# Per spacing between hands in validation
c = [df['L15_1'].values] + [df['L15_2'].values] 
c = [item for sublist in c for item in sublist]
# Per spacing between hands in NG
d = [df['nograsp_uc15cm_sp'].values]*2
d = [item for sublist in d for item in sublist]


# difference between the calc spacing between hands for validation and NG
a =  [x1 - x2 for (x1, x2) in zip(a, b)]
b =  [x1 - x2 for (x1, x2) in zip(c, d)]
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)

# Calc
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_diff], y1, 'ok', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_diff-.05], mean_a, 'ok', markersize=15, marker='^')
plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)

x_diff = x_diff + .2

# Per
for x1, y1, in zip(jit_b, b):
    plt.plot([x1+x_diff], y1, 'ok', marker='^')

mean_b = mean(b)
sd_b = std(b)
n = 30
mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
             mean_b + (sd_b / sqrt(n)) * t_val]
plt.plot([x_diff-.05], mean_b, 'ok', markersize=15, marker='^')
plt.plot([x_diff-.05, x_diff-.05], mean_b_CI, '-k', linewidth=2)




# PLOTTING difference 24cm
x_diff = 4

# Calc spacing between hands in validation
a = [df['sp_calc_uc24_1'].values] + [df['sp_calc_uc24_2'].values] 
a = [item for sublist in a for item in sublist]
# Calc spacing between hands in NG
b = [df['24_calc_spacing_No_Grasp'].values]*2
b = [item for sublist in b for item in sublist]


# Per spacing between hands in validation
c = [df['L24_1'].values] + [df['L24_2'].values] 
c = [item for sublist in c for item in sublist]
# Per spacing between hands in NG
d = [df['nograsp_uc24cm_sp'].values]*2
d = [item for sublist in d for item in sublist]


# difference between the calc spacing between hands for validation and NG
a =  [x1 - x2 for (x1, x2) in zip(a, b)]
b =  [x1 - x2 for (x1, x2) in zip(c, d)]
dat = [pd.Series(a), pd.Series(b)]
jit_a, jit_b = jitter(dat)

# Calc
for x1, y1, in zip(jit_a, a):
    plt.plot([x1+x_diff], y1, 'ok', marker='^')

mean_a = mean(a)
sd_a = std(a)
n = 30
mean_a_CI = [mean_a - (sd_a / sqrt(n)) * t_val,
             mean_a + (sd_a / sqrt(n)) * t_val]
plt.plot([x_diff-.05], mean_a, 'ok', markersize=15, marker='^')
plt.plot([x_diff-.05, x_diff-.05], mean_a_CI, '-k', linewidth=2)

x_diff = x_diff + .2

# Per
for x1, y1, in zip(jit_b, b):
    plt.plot([x1+x_diff], y1, 'ok', marker='^')

mean_b = mean(b)
sd_b = std(b)
n = 30
mean_b_CI = [mean_b - (sd_b / sqrt(n)) * t_val,
             mean_b + (sd_b / sqrt(n)) * t_val]
plt.plot([x_diff-.05], mean_b, 'ok', markersize=15, marker='^')
plt.plot([x_diff-.05, x_diff-.05], mean_b_CI, '-k', linewidth=2)



plt.plot([4.2,.8], [0,0], '--k', alpha=.5)    

# Hide the right and top spines
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
    
#Removing ticks from axes
ax1.tick_params( axis='x', which='major', bottom='off', top='off', labelbottom='on', left='off', right='off', labelsize=14)
ax1.tick_params( axis='y', which='major', direction='out', length=3, bottom='off', top='off', labelbottom='on', right='off', labelsize=14)


xticks = [1 ,1.1, 1.2, 2 ,2.1, 2.2, 3 ,3.1, 3.2, 4 ,4.1, 4.2]
labels = ['Cal','-24', 'Per', 'Cal', '-15', 'Per','Cal', '15', 'Per','Cal', '24', 'Per']
plt.xticks(xticks, labels)

# vertical alignment of xtick labels
va = [ 0, -.05, 0, 0, -.05, 0, 
      0, -.05, 0, 0, -.05, 0, ]
for z, y in zip( ax1.get_xticklabels( ), va ):
    z.set_y( y )
    
ax1.set_xlabel('')
ax1.set_ylabel('Difference in spacing measures (cm)',size=20)
plt.ylim(-16,30)


plt.title('')
plt.savefig('./Graphs/data/calc_per_sp_diff_val_vs_NG.pdf')
plt.savefig('./Graphs/data/calc_per_sp_diff_val_vs_NG.svg')

