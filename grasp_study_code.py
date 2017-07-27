# ------------------MARTY --------------------
# Not sure what program you are using, but it
# keeps on adding author info. This will keep
# on getting longer and longer and is somewhat
# ugly. See if you can disactivate this
# functionality in your editor.
# ----------------------------------------------

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

    # ------------------MARTY --------------------
    # Have you tried the code below? It does not work.
    # That is because it uses .loc(), which belongs to
    # pandas.
    # ----------------------------------------------
    '''
    age.append(sub_info.loc['Age', [1]][1])
    gender.append(sub_info.loc['Gender', [1]][1])
    handedness.append(sub_info.loc['Handedness', [1]][1])
    '''

    # ------------------MARTY --------------------
    # I am not clear why you create the dataframe here.
    # This means the dataframe is create each time
    # though the loop (i.e., for each subject).
    # ----------------------------------------------

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

    # ------------------MARTY --------------------
    # I am not clear why you calculate this in the loop.
    # Would it not be better to get all the data and
    # then calculate summary statistics?
    # ----------------------------------------------

    R24 = val.R24.mean()
    R15 = val.R15.mean()
    L24 = val.L24.mean()
    L15 = val.L15.mean()
    
    #R24 = df_ownership.grasp_c24cm_own.mean()
    #R15 = df_ownership.grasp_c15cm_own.mean()
    #L24 = df_ownership.grasp_uc24cm_own.mean()
    #L15 = df_ownership.grasp_uc15cm_own.mean()

    # ------------------MARTY --------------------
    # Not clear how this single calculation is our
    # measure of horizontal drift.
    # ----------------------------------------------
    val['horizontal diff'] = val.L15 - val.L24

    # ------------------MARTY --------------------
    # I am not clear why you are looping through this
    # file a second time. You should be able to parse
    # the entire file, line by line, in only one pass.
    # There is not nead to loop through it once for
    # the validation data and then for the grasp/
    # no-grasp data.
    # The key is keeping track of which block you are
    # currently in (Validation, grasp, no-grasp)
    # and parsing the data based on that.
    #
    # You don't ever use:
    # if current_block == 'VALIDATION':
    #       <code>
    # elif current_block == 'GRASP':
    #       <code>
    # elif current_block == 'NO_GRASP':
    #       <code>
    #
    # This is needed in order to ensure the data
    # is being stored in the correct list.
    #
    # See if you can figure this out.
    # ----------------------------------------------

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
                         grasp_uc15cm_l.append(float(ans))
                     elif cond.split("_")[4] == "24CM":
                         grasp_uc24cm_l.append(float(ans))
                     elif cond.split("_")[4] == "-15CM":
                         grasp_c15cm_l.append(float(ans))
                     else:
                         cond.split("_")[4] == "-24CM"
                         grasp_c24cm_l.append(float(ans)) 
                elif cond.split("_")[2] == "RIGHT":
                     if cond.split("_")[4] == "15CM":
                         grasp_uc15cm_r.append(float(ans))
                     elif cond.split("_")[4] == "24CM":
                         grasp_uc24cm_r.append(float(ans))
                     elif cond.split("_")[4] == "-15CM":
                         grasp_c15cm_r.append(float(ans))
                     else:
                         cond.split("_")[4] == "-24CM"
                         grasp_c24cm_r.append(float(ans))
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
print(df[0:][0:30])   

x = sum(val_L15cm)
y = len(val_L15cm)

#%%


#%% Dataframes per measure
# Now, need to do the below too. 
                 
spacing = {'L24': pd.Series(grasp_uc24cm_sp),
           'L15': pd.Series(grasp_uc15cm_sp),
           'R15': pd.Series(grasp_c15cm_sp),
           'R24': pd.Series(grasp_c24cm_sp)}
df_spacing = pd.DataFrame(spacing)


ownership = {'L24': pd.Series(grasp_uc24cm_own),
             'L15': pd.Series(grasp_uc15cm_own),
             'R15': pd.Series(grasp_c15cm_own),
             'R24': pd.Series(grasp_c24cm_own)}
df_ownership = pd.DataFrame(ownership)
    
        
left = {'R15': pd.Series(grasp_c15cm_l),
        'L15': pd.Series(grasp_uc15cm_l),
        'L24': pd.Series(grasp_uc24cm_l),
        'R24': pd.Series(grasp_c24cm_l)}
df_left = pd.DataFrame(left)

right = {'R15': pd.Series(grasp_c15cm_r),
        'L15': pd.Series(grasp_uc15cm_r),
        'L24': pd.Series(grasp_uc24cm_r),
        'R24': pd.Series(grasp_c24cm_r)}
df_right = pd.DataFrame(right)

#%%
