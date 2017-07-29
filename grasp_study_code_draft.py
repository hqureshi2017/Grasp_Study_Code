# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 11:21:29 2017

@author: h.qureshi
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 08:20:30 2017

@author: Hassan
"""

#from cumming_plot import cumming_plot
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from math import sqrt
import os

# Initialise variables
age = []
gender = []
handedness = []
grasp_c24cm_sp = []
grasp_c24cm_own = []
grasp_c24cm_r = []
grasp_c24cm_l = []
grasp_c15cm_sp = []
grasp_c15cm_own = []
grasp_c15cm_r = []
grasp_c15cm_l = []
grasp_uc15cm_sp = []
grasp_uc15cm_own = []
grasp_uc15cm_r = []
grasp_uc15cm_l = []
grasp_uc24cm_sp = []
grasp_uc24cm_own = []
grasp_uc24cm_r = []
grasp_uc24cm_l = []
nograsp_c24cm_sp = []
nograsp_c24cm_r = []
nograsp_c24cm_l = []
nograsp_c15cm_sp = []
nograsp_c15cm_r = []
nograsp_c15cm_l = []
nograsp_c15cm_sp = []
nograsp_c15cm_r = []
nograsp_c15cm_l = []
nograsp_uc24cm_sp = []
nograsp_uc24cm_r = []
nograsp_uc24cm_l = []
indexes = []
val_R24cm = []
val_R15cm = []
val_L24cm = []
val_L15cm = []


# List containing subject numbers (!! need to add others later!! )
subjects = [1, 2]
for sub_num in subjects:
    
    # Load detailed file and extract subject age, gender and handeness
    if len(str(sub_num)) == 1:
        sub = 'mock' + '0' + str(sub_num)
    else:
        sub = 'mock' + str(sub_num)
    info = os.path.join('.', 'study_data', sub, sub + '.txt')
    print(info)       


# I can't quite seem to work out how the indent works. Sometimes the above code gives
# me mock01 and mock02, and other times just mock02. What am I doing wrong?


# Thanks for pointing out the importing bit. I think I was confusing myself
# with the pandas import. The readlines method seems much easier. 

# import data for subject info
    with open(info) as f:
        sub_info = f.readlines()
        print(sub_info)
    age.append(int(sub_info[3].strip()[-2:]))
    gender.append(str(sub_info[1].strip()[-1:]))
    handedness.append(str(sub_info[2].strip()[-1:]))
    indexes.append(sub)
# I used the above method. It seems to work. Is the method below better?
'''
age.append(sub_info.loc['Age', [1]][1])
gender.append(sub_info.loc['Gender', [1]][1])
handedness.append(sub_info.loc['Handedness', [1]][1])
'''


# Import Validation data from experiment
data_long = os.path.join('.', 'study_data', sub, sub + '_data.txt')
with open(data_long) as file:
      # data = file.readlines()
      for line in file:
        if (line.split(':')[0].strip()[-22:]) == 'MEASURE_SPACING_R15CM':
            val_R15cm.append(line.split(':')[1])
        if (line.split(':')[0].strip()[-22:]) == 'MEASURE_SPACING_L15CM':
            val_L15cm.append(line.split(':')[1])
        if (line.split(':')[0].strip()[-22:]) == 'MEASURE_SPACING_L24CM':
            val_L24cm.append(line.split(':')[1])
        if (line.split(':')[0].strip()[-22:]) == 'MEASURE_SPACING_R24CM':
            val_R24cm.append(line.split(':')[1])

val = pd.DataFrame({'L24':val_L24cm,
                    'L15':val_L15cm,
                    'R15':val_R15cm,
                    'R24':val_R24cm})
print(val)

# Need to find the index for specific measures and append the respective lists
with open(data_long) as file:
    for line in file:
        if line.split(":")[0] == 'BLOCK':
            current_block = line.split(":")[1]
            # This will keep track of the current block...use it to put the data in the right list.
        elif line.split(":")[0] == 'TRIAL':
            current_trial = line.split(":")[1] # This will keep track of the current trial...use it to put the data in the right list. 
        else:
            ans = line.split(":")[1]
            cond = line.split(":")[0]
            # cond will be a string with the measurement information.
            if cond.split("_")[1] == "SPACING":
                 if cond.split("_")[2] == "15CM":
                     grasp_uc15cm_sp.append(ans)
                 elif cond.split("_")[2] == "24CM":
                     grasp_uc24cm_sp.append(ans)
                 elif cond.split("_")[2] == "-15CM":
                     grasp_c15cm_sp.append(ans)
                 else:
                     cond.split("_")[2] == "-24CM"
                     grasp_c24cm_sp.append(ans)
            elif cond.split("_")[1] == "OWNERSHIP":
                 if cond.split("_")[2] == "15CM":
                     grasp_uc15cm_own.append(ans)
                 elif cond.split("_")[2] == "24CM":
                     grasp_uc24cm_own.append(ans)
                 elif cond.split("_")[2] == "-15CM":
                     grasp_c15cm_own.append(ans)
                 else:
                     cond.split("_")[2] == "-24CM"
                     grasp_c24cm_own.append(ans)
            elif cond.split("_")[2] == "LEFT":
                 if cond.split("_")[2] == "15CM":
                     grasp_uc15cm_l.append(ans)
                 elif cond.split("_")[2] == "24CM":
                     grasp_uc24cm_l.append(ans)
                 elif cond.split("_")[2] == "-15CM":
                     grasp_c15cm_l.append(ans)
                 else:
                     cond.split("_")[2] == "-24CM"
                     grasp_c24cm_l.append(ans) 
            elif cond.split("_")[2] == "RIGHT":
                 if cond.split("_")[2] == "15CM":
                     grasp_uc15cm_r.append(ans)
                 elif cond.split("_")[2] == "24CM":
                     grasp_uc24cm_r.append(ans)
                 elif cond.split("_")[2] == "-15CM":
                     grasp_c15cm_r.append(ans)
                 else:
                     cond.split("_")[2] == "-24CM"
                     grasp_c24cm_r.append(ans) 
            else:
                False
                     
# Two problems: i. it gets both grasp and no grasp condition
# and ii. the for loop calls only on the mock02. 
# Now, need to do the below too. 
                 
            
exp1 = pd.DataFrame({'age': age,
                     'gender': gender,
                     'handedness': handedness})
print(exp1)
exp1.plot()


d = { 'age': pd.Series(age),
      'gender': pd.Series(gender),
      'handedness': pd.Series(handedness),
      'grasp_c15cm_own': pd.Series(grasp_c15cm_own),
      'grasp_c15cm_r': pd.Series(grasp_c15cm_r),
      'grasp_c15cm_l': pd.Series(grasp_c15cm_l),
      'grasp_uc15cm_sp': pd.Series(grasp_uc15cm_sp),
      'grasp_uc15cm_own': pd.Series(grasp_uc15cm_own),
      'grasp_uc15cm_r': pd.Series(grasp_uc15cm_r),
      'grasp_uc15cm_l': pd.Series(grasp_uc15cm_l),
      'grasp_uc24cm_sp': pd.Series(grasp_uc24cm_sp),
      'grasp_uc24cm_own': pd.Series(grasp_uc24cm_own),
      'grasp_uc24cm_r': pd.Series(grasp_uc24cm_r),
      'grasp_uc24cm_l': pd.Series(grasp_uc24cm_l),
      'nograsp_c24cm_sp': pd.Series(nograsp_c24cm_sp),
      'nograsp_c24cm_r': pd.Series(nograsp_c24cm_r),
      'nograsp_c24cm_l': pd.Series(nograsp_c24cm_l),
      'nograsp_c15cm_sp': pd.Series(nograsp_c15cm_sp),
      'nograsp_c15cm_r': pd.Series(nograsp_c15cm_r),
      'nograsp_c15cm_l': pd.Series(nograsp_c15cm_l),
      'nograsp_c15cm_sp': pd.Series(nograsp_c15cm_sp),
      'nograsp_c15cm_r': pd.Series(nograsp_c15cm_r),
      'nograsp_c15cm_l': pd.Series(nograsp_c15cm_l),
      'nograsp_uc24cm_sp': pd.Series(nograsp_uc24cm_sp),
      'nograsp_uc24cm_r': pd.Series(nograsp_uc24cm_r),
      'nograsp_uc24cm_l': pd.Series(nograsp_uc24cm_l),
      'val_R24cm': pd.Series(val_R24cm),
      'val_R15cm': pd.Series(val_R15cm),
      'val_L24cm': pd.Series(val_L24cm),
      'val_L15cm': pd.Series(val_L15cm)}

df = pd.DataFrame(d)   
print(df)
plt.plot(df['val_L15cm'])
