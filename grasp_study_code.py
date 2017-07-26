# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 13:48:18 2017

@author: h.qureshi
"""

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
nograsp_uc15cm_sp = []
nograsp_uc15cm_r = []
nograsp_uc15cm_l = []
nograsp_uc24cm_sp = []
nograsp_uc24cm_r = []
nograsp_uc24cm_l = []
indexes = []
val_R24cm = []
val_R15cm = []
val_L24cm = []
val_L15cm = []


# List containing subject numbers 
subjects = [1, 2, 3,4,5,7,8,9,10,
            11,12,13,14,15,16,17,18,19,20,
            21,22,23,24,25,26,27,28,29,30,31]
for sub_num in subjects:

    # Load detailed file and extract subject age, gender and handeness
    if len(str(sub_num)) == 1:
        sub = 'sub' + '0' + str(sub_num)
    else:
        sub = 'sub' + str(sub_num)
    info = os.path.join('.', 'data', sub, sub + '.txt')
    #print(info)       


    # import data for subject info
    with open(info) as f:
        sub_info = f.readlines()
    age.append(int(sub_info[3].strip()[-2:]))
    gender.append(str(sub_info[1].strip()[-1:]))
    handedness.append(str(sub_info[2].strip()[-1:]))
    indexes.append(sub)
    #    I used the above method. It seems to work. Is the method below better?
    '''
    age.append(sub_info.loc['Age', [1]][1])
    gender.append(sub_info.loc['Gender', [1]][1])
    handedness.append(sub_info.loc['Handedness', [1]][1])
    '''
    # Dataframe for the sub info            
    exp1 = pd.DataFrame({'age': age,
                         'gender': gender,
                         'handedness': handedness})

   
    # Import Validation data from experiment
    data_long = os.path.join('.', 'data', sub, sub + '_data.txt')
    
    with open(data_long) as file:
        #print(data_long)
    # data = file.readlines()
        for line in file:
            if (line.split(':')[0].strip()[-22:]) == 'MEASURE_SPACING_R15CM':
                val_R15cm.append(int(line.split(':')[1]))
            if (line.split(':')[0].strip()[-22:]) == 'MEASURE_SPACING_L15CM':
                val_L15cm.append(int(line.split(':')[1]))
            if (line.split(':')[0].strip()[-22:]) == 'MEASURE_SPACING_L24CM':
                val_L24cm.append(int(line.split(':')[1]))
            if (line.split(':')[0].strip()[-22:]) == 'MEASURE_SPACING_R24CM':
                val_R24cm.append(int(line.split(':')[1]))
    
    #DataFrame for validation data
    val = pd.DataFrame({'L24':val_L24cm,
                        'L15':val_L15cm,
                        'R15':val_R15cm,
                        'R24':val_R24cm})
    #val.plot()

    # Averages
  
    R24 = val.R24.mean()
    R15 = val.R15.mean()
    L24 = val.L24.mean()
    L15 = val.L15.mean()
    
    #R24 = df_ownership.grasp_c24cm_own.mean()
    #R15 = df_ownership.grasp_c15cm_own.mean()
    #L24 = df_ownership.grasp_uc24cm_own.mean()
    #L15 = df_ownership.grasp_uc15cm_own.mean()

    val['horizontal diff'] = val.L15 - val.L24
    

    # Two problems: i. it gets both grasp and no grasp condition !!!
    with open(data_long) as file:
        print(data_long)
        for line in file:
            if line.split(":")[0] == 'BLOCK':
                current_block = line.split(":")[1]
            elif line.split(":")[0] == 'TRIAL':
                current_trial = line.split(":")[1] 
            else:
                ans = line.split(":")[1]
                cond = line.split(":")[0]
                if cond.split("_")[1] == "SPACING":
                     if cond.split("_")[2] == "15CM":
                         grasp_uc15cm_sp.append(int(ans))
                     elif cond.split("_")[2] == "24CM":
                         grasp_uc24cm_sp.append(int(ans))
                     elif cond.split("_")[2] == "-15CM":
                         grasp_c15cm_sp.append(int(ans))
                     else:
                         cond.split("_")[2] == "-24CM"
                         grasp_c24cm_sp.append(int(ans))
                elif cond.split("_")[1] == "OWNERSHIP":
                     if cond.split("_")[2] == "15CM":
                         grasp_uc15cm_own.append(int(ans))
                     elif cond.split("_")[2] == "24CM":
                         grasp_uc24cm_own.append(int(ans))
                     elif cond.split("_")[2] == "-15CM":
                         grasp_c15cm_own.append(int(ans))
                     else:
                         cond.split("_")[2] == "-24CM"
                         grasp_c24cm_own.append(int(ans))
                elif cond.split("_")[2] == "LEFT":
                    # !! Problem. It's picking up validation location measures too
                     if cond.split("_")[4] == "15CM":
                         grasp_uc15cm_l.append(ans)
                     elif cond.split("_")[4] == "24CM":
                         grasp_uc24cm_l.append(ans)
                     elif cond.split("_")[4] == "-15CM":
                         grasp_c15cm_l.append(int(ans))
                     else:
                         cond.split("_")[4] == "-24CM"
                         grasp_c24cm_l.append(ans) 
                elif cond.split("_")[2] == "RIGHT":
                     if cond.split("_")[4] == "15CM":
                         grasp_uc15cm_r.append(ans)
                     elif cond.split("_")[4] == "24CM":
                         grasp_uc24cm_r.append(ans)
                     elif cond.split("_")[4] == "-15CM":
                         grasp_c15cm_r.append(ans)
                     else:
                         cond.split("_")[4] == "-24CM"
                         grasp_c24cm_r.append(ans) 
                else:
                    False
                     


# Dataframe for actual data
d = { 'age': pd.Series(age),
      'gender': pd.Series(gender),
      'handedness': pd.Series(handedness),
      'grasp_c15cm_sp': pd.Series(grasp_c15cm_sp),  
      'grasp_c15cm_own': pd.Series(grasp_c15cm_own),
      'grasp_c15cm_r': pd.Series(grasp_c15cm_r),
      'grasp_c15cm_l': pd.Series(grasp_c15cm_l),
      'grasp_uc15cm_sp': pd.Series(grasp_uc15cm_sp),
      'grasp_uc15cm_own': pd.Series(grasp_uc15cm_own),
      'grasp_uc15cm_r': pd.Series(grasp_uc15cm_r),
      'grasp_uc15cm_l': pd.Series(grasp_uc15cm_l),
      'grasp_c24cm_sp': pd.Series(grasp_c24cm_sp),
      'grasp_c24cm_own': pd.Series(grasp_c24cm_own),
      'grasp_c24cm_r': pd.Series(grasp_c24cm_r),
      'grasp_c24cm_l': pd.Series(grasp_c24cm_l),
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
      'nograsp_uc15cm_sp': pd.Series(nograsp_uc15cm_sp),
      'nograsp_uc15cm_r': pd.Series(nograsp_uc15cm_r),
      'nograsp_uc15cm_l': pd.Series(nograsp_uc15cm_l),
      'nograsp_uc24cm_sp': pd.Series(nograsp_uc24cm_sp),
      'nograsp_uc24cm_r': pd.Series(nograsp_uc24cm_r),
      'nograsp_uc24cm_l': pd.Series(nograsp_uc24cm_l),
      'val_R24cm': pd.Series(val_R24cm),
      'val_R15cm': pd.Series(val_R15cm),
      'val_L24cm': pd.Series(val_L24cm),
      'val_L15cm': pd.Series(val_L15cm)}
#%%
df = pd.DataFrame(d)   

x = sum(val_L15cm)
y = len(val_L15cm)

#%%


#%% Dataframes per measure
# Now, need to do the below too. 
                 
spacing = {'grasp_c15cm_sp': pd.Series(grasp_c15cm_sp),
           'grasp_uc15cm_sp': pd.Series(grasp_uc15cm_sp),
           'grasp_uc24cm_sp': pd.Series(grasp_uc24cm_sp),
           'grasp_c24cm_sp': pd.Series(grasp_c24cm_sp)}
df_spacing = pd.DataFrame(spacing)

    col = spacing.columns
    new_col = [col[2],col[3],col[8],col[0]]
    df_spacing1 = df_spacing[new_col]

ownership = {'grasp_uc15cm_own': pd.Series(grasp_uc15cm_own),
           'grasp_uc24cm_own': pd.Series(grasp_uc24cm_own),
           'grasp_c15cm_own': pd.Series(grasp_c15cm_own),
           'grasp_c24cm_own': pd.Series(grasp_c24cm_own)}
df_ownership = pd.DataFrame(ownership)
    
        R24 = df_ownership.grasp_c24cm_own.mean()
        R15 = df_ownership.grasp_c15cm_own.mean()
        L24 = df_ownership.grasp_uc24cm_own.mean()
        L15 = df_ownership.grasp_uc15cm_own.mean()


left = {'grasp_c15cm_l': pd.Series(grasp_c15cm_l),
        'grasp_uc15cm_l': pd.Series(grasp_uc15cm_l),
        'grasp_uc24cm_l': pd.Series(grasp_uc24cm_l),
        'grasp_c24cm_l': pd.Series(grasp_c24cm_l)}
df_left = pd.DataFrame(left)
#%%
