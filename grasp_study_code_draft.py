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
# -------------------------------------------------------
# Add os so that paths can be created on any file system.
# -------------------------------------------------------
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
        
# -------------------------------------------------------
# The code below is not indented, so it will only run for
# 1 subject, the last 'sub' from the above for loop.
# -------------------------------------------------------

# import data for subject info
# -------------------------------------------------------
# Using os module to create path.
# -------------------------------------------------------
info = os.path.join('..', 'study_data', sub, sub + '.txt')
print(info)
# -------------------------------------------------------
# Changed 'i' to 'sub_info'. Try using informative variable
# names. 'i' is usually reserved as an iterator and is not
# informative of the data it contains.
#
# There is an issue with the import; you use pandas,
# which returns a dataframe. If you want to do this you
# need to specify the data seperator, the fact that there
# is no header, and that col 1 are your index values.
# See code below for how to do this. I typically simply
# read the file as a simply text file and split the data
# myself (not using pandas).
# -------------------------------------------------------
with open(info) as f:
    sub_info = pd.read_csv(info, sep=":", header=None, index_col=0)
print(sub_info)

# -------------------------------------------------------
# Note sure what the following code does. There was no data
# being appended (nothing in the parentheses); and the values
# you need have not been extracted from the .txt file.
#
# I have extracted the data from the dataframe, but this
# is complicated for nothing. I would recommended reading
# in the file with the more basic read.csv or something
# similar, and:
#     1. loop over each line of text
#     2. line.split(':') and check if the result first item [0]
#        is the variable you are looking for. If so, append it.
#
# I recommend this approach because this is how you will have
# to read in the data.
# -------------------------------------------------------
age.append(sub_info.loc['Age', [1]][1])
gender.append(sub_info.loc['Gender', [1]][1])
handedness.append(sub_info.loc['Handedness', [1]][1])


# -------------------------------------------------------
# See if you can read in a data file and parse the values
# as described above.
# -------------------------------------------------------
# Import data for experiment
data_long = "..\study_data\\" + sub + '\\' + sub + '_data.txt'
with open(data_long) as file:
    data = file.readlines()
    

