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


# List containing subject numbers (!! need to add others later!! )
subjects = [1, 2]
for sub_num in subjects:

    # Load detailed file and extract subject age, gender and handeness
    if len(str(sub_num)) == 1:
        sub = 'mock' + '0' + str(sub_num)
    else:
        sub = 'mock' + str(sub_num)
    info = os.path.join('..', 'study_data', sub, sub + '.txt')
    print(info)       

-------------------------------------------------------------
# I can't quite seem to work out how the indent works. Sometimes the above code gives
# me mock01 and mock02, and other times just mock02. What am I doing wrong?
-------------------------------------------------------------

# Thanks for pointing out the importing bit. I think I was confusing myself
# with the pandas import. The readlines method seems much easier. 

# import data for subject info
    with open(info) as f:
        sub_info = f.readlines()
    print(sub_info)
    age.append(int(sub_info[6].strip()[-2:]))
    gender.append(str(sub_info[2].strip()[-1:]))
    handedness.append(str(sub_info[4].strip()[-1:]))

------------------------------------------------------------
# I used the above method. It seems to work. Is the method below better?
'''
age.append(sub_info.loc['Age', [1]][1])
gender.append(sub_info.loc['Gender', [1]][1])
handedness.append(sub_info.loc['Handedness', [1]][1])
'''
------------------------------------------------------------

# Import data from experiment
    data_long = os.path.join('..', 'study_data', sub, sub + '_data.txt')
    with open(data_long) as file:
        data = file.readlines()
    print(data_long)
    print(data)

# Need to find the index for specific measures and append the respective lists

