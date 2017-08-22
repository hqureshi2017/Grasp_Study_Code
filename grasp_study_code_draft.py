#from cumming_plot import cumming_plot
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from math import sqrt
import os
import cumming_plot
import seaborn as sns
import random

# Initialise variables
sub_id = []
age = []
gender = []
handedness = []                 # sp = Spacing
grasp_c24cm_sp = []             # c = Index fingers are crossed
grasp_c24cm_own = []            # own = Ownership
grasp_c24cm_r = []              # r = Location of Right index
grasp_c24cm_l = []              # l = Location of Left index
grasp_c15cm_sp = []
grasp_c15cm_own = []
grasp_c15cm_r = []
grasp_c15cm_l = []
grasp_uc15cm_sp = []            # uc = Index fingers are uncrossed
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
val_R24cm_val1 = []
val_R24cm_val2 = []
val_R15cm_val1 = []
val_R15cm_val2 = []
val_L24cm_val1 = []
val_L24cm_val2 = []
val_L15cm_val1 = []
val_L15cm_val2 = []


# List containing subject numbers (!! need to add others later!! )
subjects = [1, 2]
for sub_num in subjects:
    flag_R24 = False
    flag_R15 = False
    flag_L24 = False
    flag_L15 = False
    # Load detailed file and extract subject age, gender and handeness
    if len(str(sub_num)) == 1:
        sub = 'mock' + '0' + str(sub_num)
    else:
        sub = 'mock' + str(sub_num)
    sub_id.append(sub)
    info = os.path.join('.', 'study_data', sub, sub + '.txt')
    print(info)       


    # import data for subject info
    with open(info) as f:
        sub_info = f.readlines()
        print(sub_info)
    age.append(int(sub_info[3].strip()[-2:]))
    gender.append(str(sub_info[1].strip()[-1:]))
    handedness.append(str(sub_info[2].strip()[-1:]))
    indexes.append(sub)

    # Import data from experiment
    data_long = os.path.join('.', 'study_data', sub, sub + '_data.txt')
    
    
    with open(data_long) as file:
        for line in file:
            ans = line.split(":")[1]
            cond = line.split(":")[0]
            if line.split(":")[0] == 'BLOCK':
                current_block = line.split(":")[1]
            elif line.split(":")[0] == 'TRIAL':
                current_trial = line.split(":")[1]
                
                
            #   VALIDATION DATA
            if (line.split(':')[0].strip()[-22:]) == 'MEASURE_SPACING_R15CM':
                if not flag_R15:
                    val_R15cm_val1.append(int(line.split(':')[1]))
                    flag_R15 = True
                elif flag_R15:
                    val_R15cm_val2.append(int(line.split(':')[1]))
            elif (line.split(':')[0].strip()[-22:]) == 'MEASURE_SPACING_L15CM':
                if not flag_L15:
                    val_L15cm_val1.append(int(line.split(':')[1]))
                    flag_L15 = True
                elif flag_L15:
                    val_L15cm_val2.append(int(line.split(':')[1]))
            elif (line.split(':')[0].strip()[-22:]) == 'MEASURE_SPACING_L24CM':
                if not flag_L24:
                    val_L24cm_val1.append(int(line.split(':')[1]))
                    flag_L24 = True
                elif flag_L24:
                    val_L24cm_val2.append(int(line.split(':')[1]))
            elif (line.split(':')[0].strip()[-22:]) == 'MEASURE_SPACING_R24CM':
                if not flag_R24:
                    val_R24cm_val1.append(int(line.split(':')[1]))
                    flag_R24 = True
                elif flag_R24:
                    val_R24cm_val2.append(int(line.split(':')[1]))
         
            #   SPACING MEASURE
            elif (line.split(':')[0].strip()[-22:]) == 'MEASURE_SPACING_15CM':
                if current_block == ' GRASP\n':
                    grasp_uc15cm_sp.append(int(ans))
                elif current_block == ' NO GRASP\n':    
                    nograsp_uc15cm_sp.append(int(ans))
            elif (line.split(':')[0].strip()[-22:]) == 'MEASURE_SPACING_-15CM':
                if current_block == ' GRASP\n':
                    grasp_c15cm_sp.append(int(ans))
                elif current_block == ' NO GRASP\n':    
                    nograsp_c15cm_sp.append(int(ans))                
            elif (line.split(':')[0].strip()[-22:]) == 'MEASURE_SPACING_24CM':
                if current_block == ' GRASP\n':
                    grasp_uc24cm_sp.append(int(ans))
                elif current_block == ' NO GRASP\n':    
                    nograsp_uc24cm_sp.append(int(ans))
            elif (line.split(':')[0].strip()[-22:]) == 'MEASURE_SPACING_-24CM':
                if current_block == ' GRASP\n':
                    grasp_c24cm_sp.append(int(ans))
                elif current_block == ' NO GRASP\n':    
                    nograsp_c24cm_sp.append(int(ans))                
                    
            # OWNERSHIP MEASURE
            elif (line.split(':')[0].strip()[-22:]) == 'MEASURE_OWNERSHIP_15CM':
                grasp_uc15cm_own.append(int(ans))
            elif (line.split(':')[0].strip()[-23:]) == 'MEASURE_OWNERSHIP_-15CM':
                grasp_c15cm_own.append(int(ans))
            elif (line.split(':')[0].strip()[-22:]) == 'MEASURE_OWNERSHIP_24CM':
                grasp_uc24cm_own.append(int(ans))                
            elif (line.split(':')[0].strip()[-23:]) == 'MEASURE_OWNERSHIP_-24CM':
                grasp_c24cm_own.append(int(ans))
 
            # LOCATION - RIGHT
            elif (line.split(':')[0].strip()[-24:]) == 'RIGHT_INDEX FINGER_-15CM':
                if current_block == ' GRASP\n':
                    grasp_c15cm_r.append(float(ans))
                elif current_block == ' NO GRASP\n':    
                    nograsp_c15cm_r.append(float(ans))
            elif (line.split(':')[0].strip()[-23:]) == 'RIGHT_INDEX FINGER_15CM':
                if current_block == ' GRASP\n':
                    grasp_uc15cm_r.append(float(ans))
                elif current_block == ' NO GRASP\n':    
                    nograsp_uc15cm_r.append(float(ans))
            elif (line.split(':')[0].strip()[-23:]) == 'RIGHT_INDEX FINGER_24CM':
                if current_block == ' GRASP\n':
                    grasp_uc24cm_r.append(float(ans))
                elif current_block == ' NO GRASP\n':    
                    nograsp_uc24cm_r.append(float(ans))
            elif (line.split(':')[0].strip()[-24:]) == 'RIGHT_INDEX FINGER_-24CM':
                if current_block == ' GRASP\n':
                    grasp_c24cm_r.append(float(ans))
                elif current_block == ' NO GRASP\n':    
                    nograsp_c24cm_r.append(float(ans))
                    
            # LOCATION - LEFT
            elif (line.split(':')[0].strip()[-23:]) == 'LEFT_INDEX FINGER_-15CM':
                if current_block == ' GRASP\n':
                    grasp_c15cm_l.append(float(ans))
                elif current_block == ' NO GRASP\n':    
                    nograsp_c15cm_l.append(float(ans))
            elif (line.split(':')[0].strip()[-22:]) == 'LEFT_INDEX FINGER_15CM':
                if current_block == ' GRASP\n':
                    grasp_uc15cm_l.append(float(ans))
                elif current_block == ' NO GRASP\n':    
                    nograsp_uc15cm_l.append(float(ans))
            elif (line.split(':')[0].strip()[-22:]) == 'LEFT_INDEX FINGER_24CM':
                if current_block == ' GRASP\n':
                    grasp_uc24cm_l.append(float(ans))
                elif current_block == ' NO GRASP\n':    
                    nograsp_uc24cm_l.append(float(ans))
            elif (line.split(':')[0].strip()[-23:]) == 'LEFT_INDEX FINGER_-24CM':
                if current_block == ' GRASP\n':
                    grasp_c24cm_l.append(float(ans))
                elif current_block == ' NO GRASP\n':    
                    nograsp_c24cm_l.append(float(ans))
                     
            
# Dataframe for actual data
d = { 'age': age,
      'gender': gender,
      'handedness': handedness,
      'grasp_c15cm_sp': pd.Series(grasp_c15cm_sp),  
      'grasp_c15cm_own': pd.Series(grasp_c15cm_own),
      'grasp_c15cm_r': pd.Series(grasp_c15cm_r),
      'grasp_c15cm_l': pd.Series(grasp_c15cm_l),
      'grasp_uc15cm_sp': pd.Series(grasp_uc15cm_sp),
      'grasp_uc15cm_own': grasp_uc15cm_own,
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
      'L24_1':val_L24cm_val1,
        'L24_2':val_L24cm_val2,
        'L15_1':val_L15cm_val1,
        'L15_2':val_L15cm_val2,
        'R15_1':val_R15cm_val1,
        'R15_2':val_R15cm_val2,
        'R24_1':val_R24cm_val1,
        'R24_2':val_R24cm_val2}

df = pd.DataFrame(d, index = sub_id)

# Way to inspect dataframe
print(df.to_string())




#%% 
# Dataframes per measure
'''
HERE
'''
#DataFrame for validation data
absolute_val_R15_1 = []   
for val in val_R15cm_val1:
    new_val = val*(-1)
    absolute_val_R15_1.append(new_val)
absolute_val_R15_2 = []       
for val in val_R15cm_val2:
    new_val = val*(-1)
    absolute_val_R15_2.append(new_val)
absolute_val_R24_1 = []   
for val in val_R24cm_val1:
    new_val = val*(-1)
    absolute_val_R24_1.append(new_val)
absolute_val_R24_2 = []       
for val in val_R24cm_val2:
    new_val = val*(-1)
    absolute_val_R24_2.append(new_val)

df_val = pd.DataFrame({'L24_1':val_L24cm_val1,
                    'L24_2':val_L24cm_val2,
                    'L15_1':val_L15cm_val1,
                    'L15_2':val_L15cm_val2,
                    'R15_1':absolute_val_R15_1,
                    'R15_2':absolute_val_R15_2,
                    'R24_1':absolute_val_R24_1,
                    'R24_2':absolute_val_R24_2},
                        index = sub_id)

#%% 

# SPACING_GRASP
absolute_c15cm = []   
for val in grasp_c15cm_sp:
    new_val = val*(-1)
    absolute_c15cm.append(new_val)

absolute_c24cm = []   
for val in grasp_c24cm_sp:
    new_val = val*(-1)
    absolute_c24cm.append(new_val)
    
spacing_grasp = {'L24': pd.Series(grasp_uc24cm_sp),
           'L15': pd.Series(grasp_uc15cm_sp),
           'R15': pd.Series(absolute_c15cm),
           'R24': pd.Series(absolute_c24cm)}
df_spacing_grasp = pd.DataFrame(spacing_grasp)

# SPACING_NO_GRASP
absolute_c15cm = []   
for val in nograsp_c15cm_sp:
    new_val = val*(-1)
    absolute_c15cm.append(new_val)

absolute_c24cm = []   
for val in nograsp_c24cm_sp:
    new_val = val*(-1)
    absolute_c24cm.append(new_val) 
    
spacing_nograsp = {'L24': pd.Series(nograsp_uc24cm_sp),
           'L15': pd.Series(nograsp_uc15cm_sp),
           'R15': pd.Series(absolute_c15cm),
           'R24': pd.Series(absolute_c24cm)}
df_spacing_nograsp = pd.DataFrame(spacing_nograsp)
#%%
# OWNERSHIP

ownership = {'L24': pd.Series(grasp_uc24cm_own),
             'L15': pd.Series(grasp_uc15cm_own),
             'R15': pd.Series(grasp_c15cm_own),
             'R24': pd.Series(grasp_c24cm_own)}
df_ownership = pd.DataFrame(ownership)
#%%
# LOCATION_LEFT_GRASP
absolute_grasp_l_c15=[]
for i in grasp_c15cm_l:
    abc = i-36
    if abc  <= 0:
        abc=abc*(-1)
    absolute_grasp_l_c15.append(abc)
absolute_grasp_l_uc15=[]
for i in grasp_uc15cm_l:
    abc = i-36
    if abc  <= 0:
        abc=abc*(-1)
    absolute_grasp_l_uc15.append(abc)
absolute_grasp_l_c24=[]
for i in grasp_c24cm_l:
    abc = i-36
    if abc  <= 0:
        abc=abc*(-1)
    absolute_grasp_l_c24.append(abc)    
absolute_grasp_l_uc24=[]
for i in grasp_uc24cm_l:
    abc = i-36
    if abc  <= 0:
        abc=abc*(-1)
    absolute_grasp_l_uc24.append(abc)    

    
left = {'R15': pd.Series(absolute_grasp_l_c15),
        'L15': pd.Series(absolute_grasp_l_uc15),
        'L24': pd.Series(absolute_grasp_l_uc24),
        'R24': pd.Series(absolute_grasp_l_c24)}
df_left_grasp = pd.DataFrame(left)

# LOCATION_LEFT_NO_GRASP
absolute_nograsp_l_c15=[]
for i in nograsp_c15cm_l:
    abc = i-36
    if abc  <= 0:
        abc=abc*(-1)
    absolute_nograsp_l_c15.append(abc)
absolute_nograsp_l_uc15=[]
for i in nograsp_uc15cm_l:
    abc = i-36
    if abc  <= 0:
        abc=abc*(-1)
    absolute_nograsp_l_uc15.append(abc)
absolute_nograsp_l_c24=[]
for i in nograsp_c24cm_l:
    abc = i-36
    if abc  <= 0:
        abc=abc*(-1)
    absolute_nograsp_l_c24.append(abc)    
absolute_nograsp_l_uc24=[]
for i in nograsp_uc24cm_l:
    abc = i-36
    if abc  <= 0:
        abc=abc*(-1)
    absolute_nograsp_l_uc24.append(abc)    

    
left = {'R15': pd.Series(absolute_nograsp_l_c15),
        'L15': pd.Series(absolute_nograsp_l_uc15),
        'L24': pd.Series(absolute_nograsp_l_uc24),
        'R24': pd.Series(absolute_nograsp_l_c24)}
df_left_nograsp = pd.DataFrame(left)
#%%

# LOCATION_RIGHT_NO_GRASP
absolute_nograsp_r_c15=[]
for i in nograsp_c15cm_r:
    abc = i-36
    if abc  <= 0:
        abc=abc*(-1)
    absolute_nograsp_r_c15.append(abc)
absolute_nograsp_r_uc15=[]
for i in nograsp_uc15cm_r:
    abc = i-36
    if abc  <= 0:
        abc=abc*(-1)
    absolute_nograsp_r_uc15.append(abc)
absolute_nograsp_r_c24=[]
for i in nograsp_c24cm_r:
    abc = i-36
    if abc  <= 0:
        abc=abc*(-1)
    absolute_nograsp_r_c24.append(abc)    
absolute_nograsp_r_uc24=[]
for i in nograsp_uc24cm_r:
    abc = i-36
    if abc  <= 0:
        abc=abc*(-1)
    absolute_nograsp_r_uc24.append(abc) 
    
right = {'R15': pd.Series(absolute_nograsp_r_c15),
        'L15': pd.Series(absolute_nograsp_r_uc15),
        'L24': pd.Series(absolute_nograsp_r_uc24),
        'R24': pd.Series(absolute_nograsp_r_c24)}
df_right_nograsp = pd.DataFrame(right)

# LOCATION_RIGHT_GRASP
absolute_grasp_r_c15=[]
for i in grasp_c15cm_r:
    abc = i-36
    if abc  <= 0:
        abc=abc*(-1)
    absolute_grasp_r_c15.append(abc)
absolute_grasp_r_uc15=[]
for i in grasp_uc15cm_r:
    abc = i-36
    if abc  <= 0:
        abc=abc*(-1)
    absolute_grasp_r_uc15.append(abc)
absolute_grasp_r_c24=[]
for i in grasp_c24cm_r:
    abc = i-36
    if abc  <= 0:
        abc=abc*(-1)
    absolute_grasp_r_c24.append(abc)    
absolute_grasp_r_uc24=[]
for i in grasp_uc24cm_l:
    abc = i-36
    if abc  <= 0:
        abc=abc*(-1)
    absolute_grasp_r_uc24.append(abc)    

    
right = {'R15': pd.Series(absolute_grasp_r_c15),
        'L15': pd.Series(absolute_grasp_r_uc15),
        'L24': pd.Series(absolute_grasp_r_uc24),
        'R24': pd.Series(absolute_grasp_r_c24)}
df_right_grasp = pd.DataFrame(right)

            
# ------------------------------------------
#       GRAPHS
# ------------------------------------------
#                            !!! Need to make graphs on same dimensions !!!
# PERCEIVED SPACING_GRASP
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2})
fig = plt.figure(figsize=(9, 15))

blinding = True
trials = ['A', 'B', 'C', 'D']
if blinding:
    random.shuffle(trials)
    
ax1 = fig.add_subplot(211)
sns.swarmplot(data=df_spacing_grasp)
ax1.set_xlabel('Horizontal Locations')
ax1.set_ylabel('perceived spacing (cm)')
x = [0, 1, 2,3]
labels = trials
plt.xticks(x, labels, rotation='horizontal')
plt.title('SPACING_GRASP')

# PERCEIVED SPACING_NO_GRASP
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2})
fig = plt.figure(figsize=(9, 15))

blinding = True
trials = ['A', 'B', 'C', 'D']
if blinding:
    random.shuffle(trials)
    
ax1 = fig.add_subplot(211)
sns.swarmplot(data=df_spacing_nograsp)
ax1.set_xlabel('Horizontal Locations')
ax1.set_ylabel('perceived spacing (cm)')
x = [0, 1, 2,3]
labels = trials
plt.xticks(x, labels, rotation='horizontal')
plt.title('SPACING_NO_GRASP')

# PERCEIVED OWNERSHIP
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2})
fig = plt.figure(figsize=(9, 15))

blinding = True
trials = ['A', 'B', 'C', 'D']
if blinding:
    random.shuffle(trials)
    
ax1 = fig.add_subplot(211)
sns.swarmplot(data=df_ownership)
ax1.set_xlabel('Horizontal Locations')
ax1.set_ylabel('perceived Ownership')
x = [0, 1, 2,3]
labels = trials
plt.xticks(x, labels, rotation='horizontal')
plt.title('OWNERSHIP')


# LOCATION_LEFT_GRASP_15cm
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2})
fig = plt.figure(figsize=(9, 15))

blinding = True
trials = ['A', 'B', 'C', 'D']
if blinding:
    random.shuffle(trials)
    
ax1 = fig.add_subplot(211)
sns.swarmplot(data=df_left_grasp)
ax1.set_xlabel('Horizontal Locations')
ax1.set_ylabel('Perceived location (cm)')
x = [0, 1, 2,3]
labels = trials
plt.xticks(x, labels, rotation='horizontal')
plt.title('Location LEFT GRASP')


# LOCATION_LEFT_NO_GRASP_15cm
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2})
fig = plt.figure(figsize=(9, 15))

blinding = True
trials = ['A', 'B', 'C', 'D']
if blinding:
    random.shuffle(trials)
    
ax1 = fig.add_subplot(211)
sns.swarmplot(data=df_left_nograsp)
ax1.set_xlabel('Horizontal Locations')
ax1.set_ylabel('Perceived location (cm)')
x = [0, 1, 2,3]
labels = trials
plt.xticks(x, labels, rotation='horizontal')
plt.title('Location LEFT NO GRASP')

# LOCATION_RIGHT_GRASP_15cm
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2})
fig = plt.figure(figsize=(9, 15))

blinding = True
trials = ['A', 'B', 'C', 'D']
if blinding:
    random.shuffle(trials)
    
ax1 = fig.add_subplot(211)
sns.swarmplot(data=df_right_grasp)
ax1.set_xlabel('Horizontal Locations')
ax1.set_ylabel('Perceived location (cm)')
x = [0, 1, 2,3]
labels = trials
plt.xticks(x, labels, rotation='horizontal')
plt.title('Location RIGHT GRASP')


# LOCATION_RIGHT_NO_GRASP_15cm
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2})
fig = plt.figure(figsize=(9, 15))

blinding = True
trials = ['A', 'B', 'C', 'D']
if blinding:
    random.shuffle(trials)
    
ax1 = fig.add_subplot(211)
sns.swarmplot(data=df_right_nograsp)
ax1.set_xlabel('Horizontal Locations')
ax1.set_ylabel('Perceived location (cm)')
x = [0, 1, 2,3]
labels = trials
plt.xticks(x, labels, rotation='horizontal')
plt.title('Location RIGHT NO GRASP')



#%%
# VALIDATION 
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2})
fig = plt.figure(figsize=(9, 15))

blinding = True
trials = ['A', 'B', 'C', 'D', 'E', 'F','G','H']
if blinding:
    random.shuffle(trials)
    
ax1 = fig.add_subplot(211)
sns.swarmplot(data=df_val)
ax1.set_xlabel('Horizontal Locations')
ax1.set_ylabel('Perceived Spacing (cm)')
x = [0, 1, 2,3,4,5,6,7]
labels = trials
plt.xticks(x, labels, rotation='horizontal')
plt.title('VALIDATION')
#plt.savefig('test.pdf')

#--------------------------------------
#PAIRED GRAPHS
#--------------------------------------
# Ownership_distance_uc
uncrossed = [grasp_uc15cm_own, grasp_uc24cm_own]
fig = plt.figure()
ax = fig.add_subplot(111)
paired(uncrossed, ax, likert=True)
plt.suptitle('Example with Likert scale, close to see next...')
plt.show()

# Ownership_distance_c
crossed = [grasp_c15cm_own, grasp_c24cm_own]
fig = plt.figure()
ax = fig.add_subplot(111)
paired(crossed, ax, likert=True)
plt.suptitle('Example with Likert scale, close to see next...')
plt.show()

# Ownership_crossed_24
crossed_24 = [grasp_uc24cm_own,grasp_c24cm_own]
fig = plt.figure()
ax = fig.add_subplot(111)
paired(crossed_24, ax, likert=True)
plt.suptitle('Example with Likert scale, close to see next...')
plt.show()

# Ownership_crossed_15
crossed_15 = [grasp_uc15cm_own,grasp_c15cm_own]
fig = plt.figure()
ax = fig.add_subplot(111)
paired(crossed_15, ax, likert=True)
plt.suptitle('Example with Likert scale, close to see next...')
plt.show()
