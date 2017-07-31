

# ------------------MARTY --------------------
# Not sure what program you are using, but it
# keeps on adding author info. This will keep
# on getting longer and longer and is somewhat
# ugly. See if you can disactivate this
# functionality in your editor.
# 
#----------------- Hassan -----------------------
# Using Spyder. Not sure how to get rid of it.  


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
# TODO: Add conditionals for validation data
val_R24cm_val1 = []
val_R24cm_val2 = []
val_R15cm_val1 = []
val_R15cm_val2 = []
val_L24cm_val1 = []
val_L24cm_val2 = []
val_L15cm_val1 = []
val_L15cm_val2 = []


# List containing subject numbers 
subjects = [1, 2, 3,4,5,7,8,9,10,
            11,12,13,14,15,16,17,18,19,20,
            21,22,23,24,25,26,27,28,29,30,31]
for sub_num in subjects:
    flag_R24 = False
    flag_R15 = False
    flag_L24 = False
    flag_L15 = False
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
    
    
    # Import data from experiment
    data_long = os.path.join('.', 'data', sub, sub + '_data.txt')
    
    
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
                val_R15cm.append(int(line.split(':')[1]))
            elif (line.split(':')[0].strip()[-22:]) == 'MEASURE_SPACING_L15CM':
                val_L15cm.append(int(line.split(':')[1]))
            elif (line.split(':')[0].strip()[-22:]) == 'MEASURE_SPACING_L24CM':
                val_L24cm.append(int(line.split(':')[1]))
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
         
                    
# Dataframe for the sub info
# TODO:  index rows with sub id
exp1 = pd.DataFrame({'age': age,
                         'gender': gender,
                         'handedness': handedness})
                    
#DataFrame for validation data
# TODO: Make dataframe with 30 rows and 4 columns; index rows with sub id
val = pd.DataFrame({'L24':val_L24cm,
                        'L15':val_L15cm,
                        'R15':val_R15cm,
                        'R24':val_R24cm})

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
      #'val_R24cm': pd.Series(val_R24cm),
      #'val_R15cm': pd.Series(val_R15cm),
      #'val_L24cm': pd.Series(val_L24cm),
      #'val_L15cm': pd.Series(val_L15cm)
      }
# TODO:  index rows with sub id
df = pd.DataFrame(d)

# Way to inspect dataframe
print(df.to_string())

# Convert gender and handedness to category
df['gender'] = df['gender'].astype('category')
df['handedness'] = df['handedness'].astype('category')

# Calculate difference scores
df['24cm_grasp_vs_nograsp_sp'] = df.grasp_uc24cm_sp - df.nograsp_uc24cm_sp
df['24cm_grasp_vs_nograsp_r'] = df.grasp_uc24cm_r - df.nograsp_uc24cm_r
df['24cm_grasp_vs_nograsp_l'] = df.grasp_uc24cm_l - df.nograsp_uc24cm_l

df['15cm_grasp_vs_nograsp_sp'] = df.grasp_uc15cm_sp - df.nograsp_uc15cm_sp
df['15cm_grasp_vs_nograsp_r'] = df.grasp_uc15cm_r - df.nograsp_uc15cm_r
df['15cm_grasp_vs_nograsp_l'] = df.grasp_uc15cm_l - df.nograsp_uc15cm_l

df['-15cm_grasp_vs_nograsp_sp'] = df.grasp_c15cm_sp - df.nograsp_c15cm_sp
df['-15cm_grasp_vs_nograsp_r'] = df.grasp_c15cm_r - df.nograsp_c15cm_r
df['-15cm_grasp_vs_nograsp_l'] = df.grasp_c15cm_l - df.nograsp_c15cm_l

df['-24cm_grasp_vs_nograsp_sp'] = df.grasp_c24cm_sp - df.nograsp_c24cm_sp
df['-24cm_grasp_vs_nograsp_r'] = df.grasp_c24cm_r - df.nograsp_c24cm_r
df['-24cm_grasp_vs_nograsp_l'] = df.grasp_c24cm_l - df.nograsp_c24cm_l





# Dataframes per measure
                 
spacing = {'L24': pd.Series(grasp_uc24cm_sp),
           'L15': pd.Series(grasp_uc15cm_sp),
           'R15': pd.Series(grasp_c15cm_sp),
           'R24': pd.Series(grasp_c24cm_sp)}
df_spacing = pd.DataFrame(spacing)

spacing_nograsp = {'L24': pd.Series(nograsp_uc24cm_sp),
           'L15': pd.Series(nograsp_uc15cm_sp),
           'R15': pd.Series(nograsp_c15cm_sp),
           'R24': pd.Series(nograsp_c24cm_sp)}
df_spacing_nograsp = pd.DataFrame(spacing_nograsp)


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
# ------------------------------------------
# Stats
# ------------------------------------------                
   
for i in df:
    if df[i].dtypes == 'float64':
        print ('{:>20}  count = {:<2.0f}  mean = {:>5.2f}  SD = {:>5.2f}  95%CI = {:>3.2f} min = {:>3.0f}  max = {:>3.0f}'.
               format(i, df[i].count(), df[i].mean(), df[i].std(), (df[i].std()/sqrt(len(df[i])))*1.96, df[i].min(), df[i].max()))




    R24 = val.R24.mean()
    R15 = val.R15.mean()
    L24 = val.L24.mean()
    L15 = val.L15.mean()
    
    #R24 = df_ownership.grasp_c24cm_own.mean()
    #R15 = df_ownership.grasp_c15cm_own.mean()
    #L24 = df_ownership.grasp_uc24cm_own.mean()
    #L15 = df_ownership.grasp_uc15cm_own.mean()
    
#%%


#   !! Need to plot !!