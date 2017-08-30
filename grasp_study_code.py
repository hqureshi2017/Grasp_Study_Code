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
from collections import OrderedDict
from scipy.stats import t

########### HASSAN ##############
#   Code contains the Dataframe for the Grasp Study Data.
#    Also has the calculations. Some of these were put on a new Dataframe due to bigger size
#   See figures.py for more upto date Graphs.
#   I've added the Analysis below just to test out, but it still needs work. 
################################## 
   
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

val_loc_R_R24cm_val1 = []
val_loc_R_R24cm_val2 = []
val_loc_R_R15cm_val1 = []
val_loc_R_R15cm_val2 = []
val_loc_R_L24cm_val1 = []
val_loc_R_L24cm_val2 = []
val_loc_R_L15cm_val1 = []
val_loc_R_L15cm_val2 = []


val_loc_L_R24cm_val1 = []
val_loc_L_R24cm_val2 = []
val_loc_L_R15cm_val1 = []
val_loc_L_R15cm_val2 = []
val_loc_L_L24cm_val1 = []
val_loc_L_L24cm_val2 = []
val_loc_L_L15cm_val1 = []
val_loc_L_L15cm_val2 = []

# List containing subject numbers (!! need to add others later!! )
subjects = [1, 2, 3, 4,5,7,8,9,10,
            11,12, 13,14,15,16,17,18,19,20,
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
    sub_id.append(sub)
    info = os.path.join('.', 'data', sub, sub + '.txt')


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
                
            #   VALIDATION DATA_SPACING
            if (line.split(':')[0].strip()[-22:]) == 'MEASURE_SPACING_R15CM':
                if not flag_R15:
                    val_R15cm_val1.append(int(line.split(':')[1]))
                    flag_R15 = True
                elif flag_R15:
                    val_R15cm_val2.append(int(line.split(':')[1]))
                    flag_R15 = False
            elif (line.split(':')[0].strip()[-22:]) == 'MEASURE_SPACING_L15CM':
                if not flag_L15:
                    val_L15cm_val1.append(int(line.split(':')[1]))
                    flag_L15 = True
                elif flag_L15:
                    val_L15cm_val2.append(int(line.split(':')[1]))
                    flag_L15 = False
            elif (line.split(':')[0].strip()[-22:]) == 'MEASURE_SPACING_L24CM':
                if not flag_L24:
                    val_L24cm_val1.append(int(line.split(':')[1]))
                    flag_L24 = True
                elif flag_L24:
                    val_L24cm_val2.append(int(line.split(':')[1]))
                    flag_L24 = False
            elif (line.split(':')[0].strip()[-22:]) == 'MEASURE_SPACING_R24CM':
                if not flag_R24:
                    val_R24cm_val1.append(int(line.split(':')[1]))
                    flag_R24 = True
                elif flag_R24:
                    val_R24cm_val2.append(int(line.split(':')[1]))
                    flag_R24 = False
         
            # VALIDATION_loc_RIGHT
            elif (line.split(':')[0].strip()[-24:]) == 'RIGHT_INDEX FINGER_R15CM':
                if not flag_R15:
                    val_loc_R_R15cm_val1.append(float(line.split(':')[1]))
                    flag_R15 = True
                elif flag_R15:
                    val_loc_R_R15cm_val2.append(float(line.split(':')[1]))
                    flag_R15 = False
            elif (line.split(':')[0].strip()[-24:]) == 'RIGHT_INDEX FINGER_R24CM':
                if not flag_R24:
                    val_loc_R_R24cm_val1.append(float(line.split(':')[1]))
                    flag_R24 = True
                elif flag_R24:
                    val_loc_R_R24cm_val2.append(float(line.split(':')[1]))
                    flag_R24 = False
            elif (line.split(':')[0].strip()[-24:]) == 'RIGHT_INDEX FINGER_L15CM':
                if not flag_L15:
                    val_loc_R_L15cm_val1.append(float(line.split(':')[1]))
                    flag_L15 = True
                elif flag_L15:
                    val_loc_R_L15cm_val2.append(float(line.split(':')[1]))
                    flag_L15 = False
            elif (line.split(':')[0].strip()[-24:]) == 'RIGHT_INDEX FINGER_L24CM':
                if not flag_L24:
                    val_loc_R_L24cm_val1.append(float(line.split(':')[1]))
                    flag_L24 = True
                elif flag_L24:
                    val_loc_R_L24cm_val2.append(float(line.split(':')[1]))
                    flag_L24 = False
                    
            # VALIDATION_loc_LEFT
            elif (line.split(':')[0].strip()[-23:]) == 'LEFT_INDEX FINGER_R15CM':
                if not flag_R15:
                    val_loc_L_R15cm_val1.append(float(line.split(':')[1]))
                    flag_R15 = True
                elif flag_R15:
                    val_loc_L_R15cm_val2.append(float(line.split(':')[1]))
                    flag_R15 = False
            elif (line.split(':')[0].strip()[-23:]) == 'LEFT_INDEX FINGER_R24CM':
                if not flag_R24:
                    val_loc_L_R24cm_val1.append(float(line.split(':')[1]))
                    flag_R24 = True
                elif flag_R24:
                    val_loc_L_R24cm_val2.append(float(line.split(':')[1]))
                    flag_R24 = False
            elif (line.split(':')[0].strip()[-23:]) == 'LEFT_INDEX FINGER_L15CM':
                if not flag_L15:
                    val_loc_L_L15cm_val1.append(float(line.split(':')[1]))
                    flag_L15 = True
                elif flag_L15:
                    val_loc_L_L15cm_val2.append(float(line.split(':')[1]))
                    flag_L15 = False
            elif (line.split(':')[0].strip()[-23:]) == 'LEFT_INDEX FINGER_L24CM':
                if not flag_L24:
                    val_loc_L_L24cm_val1.append(float(line.split(':')[1]))
                    flag_L24 = True
                elif flag_L24:
                    val_loc_L_L24cm_val2.append(float(line.split(':')[1]))
                    flag_L24 = False
                    
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
                grasp_uc15cm_own.append(int(ans)+1)
            elif (line.split(':')[0].strip()[-23:]) == 'MEASURE_OWNERSHIP_-15CM':
                grasp_c15cm_own.append(int(ans)+1)
            elif (line.split(':')[0].strip()[-22:]) == 'MEASURE_OWNERSHIP_24CM':
                grasp_uc24cm_own.append(int(ans)+1)                
            elif (line.split(':')[0].strip()[-23:]) == 'MEASURE_OWNERSHIP_-24CM':
                grasp_c24cm_own.append(int(ans)+1)
 
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
        
        # NOTE: Changed Ownership values to +1 for new range 1-7
      
            
# Dataframe for actual data
d = OrderedDict({ 'age': age,
      'gender': gender,
      'handedness': handedness,
      'grasp_c15cm_sp': pd.Series(grasp_c15cm_sp, index = sub_id),
      'grasp_c15cm_own': pd.Series(grasp_c15cm_own, index = sub_id),
      'grasp_c15cm_r': pd.Series(grasp_c15cm_r, index = sub_id),
      'grasp_c15cm_l': pd.Series(grasp_c15cm_l, index = sub_id),
      'grasp_uc15cm_sp': pd.Series(grasp_uc15cm_sp, index = sub_id),
      'grasp_uc15cm_own': pd.Series(grasp_uc15cm_own, index = sub_id),
      'grasp_uc15cm_r': pd.Series(grasp_uc15cm_r, index = sub_id),
      'grasp_uc15cm_l': pd.Series(grasp_uc15cm_l, index = sub_id),
      'grasp_c24cm_sp': pd.Series(grasp_c24cm_sp, index = sub_id),
      'grasp_c24cm_own': pd.Series(grasp_c24cm_own, index = sub_id),
      'grasp_c24cm_r': pd.Series(grasp_c24cm_r, index = sub_id),
      'grasp_c24cm_l': pd.Series(grasp_c24cm_l, index = sub_id),
      'grasp_uc24cm_sp': pd.Series(grasp_uc24cm_sp, index = sub_id),
      'grasp_uc24cm_own': pd.Series(grasp_uc24cm_own, index = sub_id),
      'grasp_uc24cm_r': pd.Series(grasp_uc24cm_r, index = sub_id),
      'grasp_uc24cm_l': pd.Series(grasp_uc24cm_l, index = sub_id),
      'nograsp_c24cm_sp': pd.Series(nograsp_c24cm_sp, index = sub_id),
      'nograsp_c24cm_r': pd.Series(nograsp_c24cm_r, index = sub_id),
      'nograsp_c24cm_l': pd.Series(nograsp_c24cm_l, index = sub_id),
      'nograsp_c15cm_sp': pd.Series(nograsp_c15cm_sp, index = sub_id),
      'nograsp_c15cm_r': pd.Series(nograsp_c15cm_r, index = sub_id),
      'nograsp_c15cm_l': pd.Series(nograsp_c15cm_l, index = sub_id),
      'nograsp_uc15cm_sp': pd.Series(nograsp_uc15cm_sp, index = sub_id),
      'nograsp_uc15cm_r': pd.Series(nograsp_uc15cm_r, index = sub_id),
      'nograsp_uc15cm_l': pd.Series(nograsp_uc15cm_l, index = sub_id),
      'nograsp_uc24cm_sp': pd.Series(nograsp_uc24cm_sp, index = sub_id),
      'nograsp_uc24cm_r': pd.Series(nograsp_uc24cm_r, index = sub_id),
      'nograsp_uc24cm_l': pd.Series(nograsp_uc24cm_l, index = sub_id),
      'L24_1':val_L24cm_val1,
        'L24_2':val_L24cm_val2,
        'L15_1':val_L15cm_val1,
        'L15_2':val_L15cm_val2,
        'R15_1':val_R15cm_val1,
        'R15_2':val_R15cm_val2,
        'R24_1':val_R24cm_val1,
        'R24_2':val_R24cm_val2,
        
        'loc_R_L24_1':val_loc_R_L24cm_val1,
        'loc_R_L24_2':val_loc_R_L24cm_val2,
        'loc_R_L15_1':val_loc_R_L15cm_val1,
        'loc_R_L15_2':val_loc_R_L15cm_val2,
        'loc_R_R15_1':val_loc_R_R15cm_val1,
        'loc_R_R15_2':val_loc_R_R15cm_val2,
        'loc_R_R24_1':val_loc_R_R24cm_val1,
        'loc_R_R24_2':val_loc_R_R24cm_val2,
        
        'loc_L_L24_1':val_loc_L_L24cm_val2,
        'loc_L_L24_2':val_loc_L_L24cm_val1,
        'loc_L_L15_1':val_loc_L_L15cm_val2,
        'loc_L_L15_2':val_loc_L_L15cm_val1,
        'loc_L_R15_1':val_loc_L_R15cm_val2,
        'loc_L_R15_2':val_loc_L_R15cm_val1,
        'loc_L_R24_1':val_loc_L_R24cm_val2,
        'loc_L_R24_2':val_loc_L_R24cm_val1})

df = pd.DataFrame(d, index = sub_id)




# Way to inspect dataframe
print(df.to_string())

#def calc(df):
    
# Convert gender and handedness to category
df['gender'] = df['gender'].astype('category')
df['handedness'] = df['handedness'].astype('category')

# Calculate spacing difference scores
df['24cm_grasp_vs_nograsp_sp'] =  df.nograsp_uc24cm_sp - df.grasp_uc24cm_sp
df['24cm_grasp_vs_nograsp_r'] = df.nograsp_uc24cm_r - df.grasp_uc24cm_r
df['24cm_grasp_vs_nograsp_l'] = df.nograsp_uc24cm_l - df.grasp_uc24cm_l

df['15cm_grasp_vs_nograsp_sp'] = df.nograsp_uc15cm_sp - df.grasp_uc15cm_sp
df['15cm_grasp_vs_nograsp_r'] = df.nograsp_uc15cm_r - df.grasp_uc15cm_r
df['15cm_grasp_vs_nograsp_l'] = df.nograsp_uc15cm_l - df.grasp_uc15cm_l

df['-15cm_grasp_vs_nograsp_sp'] = df.nograsp_c15cm_sp - df.grasp_c15cm_sp
df['-15cm_grasp_vs_nograsp_r'] = df.nograsp_c15cm_r - df.grasp_c15cm_r
df['-15cm_grasp_vs_nograsp_l'] = df.nograsp_c15cm_l - df.grasp_c15cm_l

df['-24cm_grasp_vs_nograsp_sp'] = df.nograsp_c24cm_sp - df.grasp_c24cm_sp
df['-24cm_grasp_vs_nograsp_r'] = df.nograsp_c24cm_r - df.grasp_c24cm_r
df['-24cm_grasp_vs_nograsp_l'] = df.nograsp_c24cm_l - df.grasp_c24cm_l

# Effect of Distance + Midline on spacing
df['-15_vs_-24_sp'] = df.grasp_c15cm_sp - df.grasp_c24cm_sp
df['15_vs_24_sp'] = df.grasp_uc15cm_sp - df.grasp_uc24cm_sp
df['-15_vs_15_sp'] = df.grasp_c15cm_sp*-1 - df.grasp_uc15cm_sp
df['-24_vs_24_sp'] = df.grasp_c24cm_sp*-1 - df.grasp_uc24cm_sp

# Effect of Distance + Midline on ownership
df['-15_vs_-24_own'] = df.grasp_c15cm_own - df.grasp_c24cm_own
df['15_vs_24_own'] = df.grasp_uc15cm_own - df.grasp_uc24cm_own
df['-15_vs_15_own'] = df.grasp_c15cm_own - df.grasp_uc15cm_own
df['-24_vs_24_own'] = df.grasp_c24cm_own - df.grasp_uc24cm_own

# Calculation Spacing
df['-24_calc_spacing_Grasp'] = df.grasp_c24cm_l - df.grasp_c24cm_r
df['-15_calc_spacing_Grasp'] = df.grasp_c15cm_l - df.grasp_c15cm_r
df['15_calc_spacing_Grasp'] = df.grasp_uc15cm_l - df.grasp_uc15cm_r
df['24_calc_spacing_Grasp'] = df.grasp_uc24cm_l - df.grasp_uc24cm_r

df['-24_calc_spacing_No_Grasp'] = df.nograsp_c24cm_l - df.nograsp_c24cm_r
df['-15_calc_spacing_No_Grasp'] = df.nograsp_c15cm_l - df.nograsp_c15cm_r
df['15_calc_spacing_No_Grasp'] = df.nograsp_uc15cm_l - df.nograsp_uc15cm_r
df['24_calc_spacing_No_Grasp'] = df.nograsp_uc24cm_l - df.nograsp_uc24cm_r

df['-24_calc_sp_diff'] = df['-24_calc_spacing_No_Grasp'] - df['-24_calc_spacing_Grasp']
df['-15_calc_sp_diff'] = df['-15_calc_spacing_No_Grasp'] - df['-15_calc_spacing_Grasp']
df['15_calc_sp_diff'] = df['15_calc_spacing_No_Grasp'] - df['15_calc_spacing_Grasp']
df['24_calc_sp_diff'] = df['24_calc_spacing_No_Grasp'] - df['24_calc_spacing_Grasp']

calc_sp = [df['-24_calc_spacing_Grasp'].values] + [df['-15_calc_spacing_Grasp'].values] + [df['15_calc_spacing_Grasp'].values] + [df['24_calc_spacing_Grasp'].values]+[df['-24_calc_spacing_No_Grasp'].values] + [df['-15_calc_spacing_No_Grasp'].values] + [df['15_calc_spacing_No_Grasp'].values] + [df['24_calc_spacing_No_Grasp'].values]
perc_sp = [df['grasp_c24cm_sp'].values] + [df['grasp_c15cm_sp'].values]+[df['grasp_uc15cm_sp'].values] + [df['grasp_uc24cm_sp'].values]+[df['nograsp_c24cm_sp'].values] + [df['nograsp_c15cm_sp'].values] + [df['nograsp_uc15cm_sp'].values]  + [df['nograsp_uc24cm_sp'].values]


df1 = pd.DataFrame()   # NEW DATAFRAME FOR VALIDATION DATA
# Perceived Spacing
df1['sp_perceived_c24'] = [df['R24_1'].values] + [df['R24_2'].values]
df1['sp_perceived_c15'] = [df['R15_1'].values] + [df['R15_2'].values]
df1['sp_perceived_uc24'] = [df['L24_1'].values] + [df['L24_2'].values]
df1['sp_perceived_uc15'] = [df['L15_1'].values] + [df['L15_2'].values]

# Calculated Spacing
df['sp_calc_c24_1'] = (df.loc_L_R24_1-36) - (df.loc_R_R24_1-36)
df['sp_calc_c24_2'] = (df.loc_L_R24_2-36) - (df.loc_R_R24_2-36)
df['sp_calc_c15_1'] = (df.loc_L_R15_1-36) - (df.loc_R_R15_1-36)
df['sp_calc_c15_2'] = (df.loc_L_R15_2-36) - (df.loc_R_R15_2-36)
df['sp_calc_uc24_1'] = (df.loc_L_L24_1-36) - (df.loc_R_L24_1-36)
df['sp_calc_uc24_2'] = (df.loc_L_L24_2-36) - (df.loc_R_L24_2-36)
df['sp_calc_uc15_1'] = (df.loc_L_L15_1-36) - (df.loc_R_L15_1-36)
df['sp_calc_uc15_2'] = (df.loc_L_L15_2-36) - (df.loc_R_L15_2-36)


# Adding the calc lists
df1['sp_calc_c24_total'] = [df['sp_calc_c24_1'].values] + [df['sp_calc_c24_2'].values]
df1['sp_calc_c15_total'] = [df['sp_calc_c15_1'].values] + [df['sp_calc_c15_2'].values]
df1['sp_calc_uc24_total'] = [df['sp_calc_uc24_1'].values] + [df['sp_calc_uc24_2'].values]
df1['sp_calc_uc15_total'] = [df['sp_calc_uc15_1'].values] + [df['sp_calc_uc15_2'].values]

# Calculate difference scores

df1['val_-24_sp_diff'] = df1['sp_calc_c24_total'] - df1['sp_perceived_c24']*-1 
df1['val_-15_sp_diff'] = df1['sp_calc_c15_total'] - df1['sp_perceived_c15']*-1 
df1['val_24_sp_diff'] = df1['sp_calc_uc24_total']*-1 - df1['sp_perceived_uc24'] 
df1['val_15_sp_diff'] = df1['sp_calc_uc15_total']*-1 - df1['sp_perceived_uc15'] 

    
#%%


'''
# -----------------------------------------
#       GRAPHS
# ------------------------------------------

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2})

# ---------------------------------------------------------------
# PLOTTING PERCEIVED SPACING AND OWNERSHIP FOR NO-GRASP AND GRASP
# ---------------------------------------------------------------

# SPACING 
fig = plt.figure(figsize=(20, 15))

blinding = True
trials = ['24cm', '15cm', '-24cm', '-15cm']

ax1 = fig.add_subplot(221) # NO-GRASP SPACING

dat = df[['nograsp_uc24cm_sp', 'nograsp_uc15cm_sp', 'nograsp_c24cm_sp', 'nograsp_c15cm_sp']].copy()
dat['nograsp_c24cm_sp'] = dat['nograsp_c24cm_sp'] * -1
dat['nograsp_c15cm_sp'] = dat['nograsp_c15cm_sp'] * -1

sns.swarmplot(data=dat)
ax1.set_xlabel('Horizontal Locations')
ax1.set_ylabel('Perceived Spacing (cm)')
x = [0, 1, 2, 3]

labels = trials
plt.xticks(x, labels, rotation='horizontal')
plt.title('A')
plt.ylim(0,40)

ax2 = fig.add_subplot(2,2,2) # GRASP SPACING

dat = df[['grasp_uc24cm_sp', 'grasp_uc15cm_sp', 'grasp_c24cm_sp', 'grasp_c15cm_sp']].copy()
dat['grasp_c24cm_sp'] = dat['grasp_c24cm_sp'] * -1
dat['grasp_c15cm_sp'] = dat['grasp_c15cm_sp'] * -1

sns.swarmplot(data=dat)
ax2.set_xlabel('Horizontal Locations')
ax2.set_ylabel('')
x = [0, 1, 2, 3]
labels = trials
plt.xticks(x, labels, rotation='horizontal')
plt.title('B')
plt.ylim(0,40)
plt.savefig('./Graphs/Spacing.pdf')


# OWNERSHIP
fig = plt.figure(figsize=(5, 5))
ax1 = fig.add_subplot(111) # OWNERSHIP
dat = df[['grasp_uc24cm_own', 'grasp_uc15cm_own', 'grasp_c24cm_own', 'grasp_c15cm_own']].copy()
sns.swarmplot(data=dat)
ax1.set_xlabel('Horizontal Locations')
ax1.set_ylabel('Perceived Ownership')
x = [0, 1, 2, 3]
labels = '24', '15', '-24', '-15'
plt.xticks(x, labels, rotation='horizontal')
plt.title('OWNERSHIP')
plt.savefig('./Graphs/Ownership.pdf')

# LOCATION MEASURES
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
plt.title('Grasp')
plt.ylim(0,32)
    
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
plt.title('No Grasp')
plt.ylim(0,32)

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
ax3.set_ylabel('')
x = [0, 1, 2,3]
plt.xticks(x, labels, rotation='horizontal')
plt.title('4')
plt.ylim(-10,15)
    
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
plt.ylim(-10,15)
plt.savefig('./Graphs/Location.pdf')


# VALIDATION

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
plt.ylim(-20,20)        

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
plt.ylim(-5,35)               



plt.savefig('./Graphs/Validation.pdf')
'''
  
####################################################
# CALCULATE DIFFERENCE VALUES AND SUMMARY STATISTICS
####################################################
'''
#!!! NOTE _ NEED TO EDIT THE BELOW CODE


# Calculate difference scores
df['24cm_grasp_vs_nograsp_sp'] =  df.nograsp_uc24cm_sp - df.grasp_uc24cm_sp
df['15m_grasp_vs_nograsp_sp'] =  df.nograsp_uc15cm_sp - df.grasp_uc15cm_sp
df['-24cm_grasp_vs_nograsp_sp'] =  df.nograsp_c24cm_sp - df.grasp_c24cm_sp
df['-15cm_grasp_vs_nograsp_sp'] =  df.nograsp_c15cm_sp - df.grasp_c15cm_sp


# Loop through items in dataframe and calculate basic summary statistics
txt = ['{:<15} {}'.format('Male', sum(df.gender == 'M')),
       '{:<15} {}'.format('Female', sum(df.gender == 'F')),
       '{:<15} {}'.format('Right handed', sum(df.handedness == 'R')),
       '{:<15} {}'.format('Left handed', sum(df.handedness == 'L')),
       '{:<15} mean = {:>4.1f}   SD = {:>3.1f}   min = {:>2.0f}   '
       'max = {:>2.0f}'.format('age', df['age'].mean(),
                               df['age'].std(), df['age'].min(), df['age'].max())]

for line in txt:
    print(line)

for loop, line in enumerate(txt):
    if loop == 0:
        flag = 'w'
    else:
        flag = 'a'
    with open('./Results/data_results.txt', flag) as file:
        file.write(line)
        file.write('\n')

with open('./Results/data_results.txt', flag) as file:
    file.write('\n')
    file.write('=' * 7)
    file.write('\nRESULTS\n')
    file.write('=' * 7)
    file.write('\n' * 2)

# Loop through items in dataframe and calculate basic summary statistics
t_val = float(t.ppf([0.975], 30))
for loop, i in enumerate(df):
    if df[i].dtypes == 'int64' or df[i].dtypes == 'float64':
        if not i == 'age':
            txt = '{:>30}  count = {:<2.0f}  mean = {:>5.2f}  SD = {:>5.2f}  95% MoE = {:>5.2f}   ' \
                  '95%CI = {:>5.2f} to {:>5.2f}   min = {:>3.0f}   ' \
                  'max = {:>3.0f}'.format(i,
                                          df[i].count(),
                                          df[i].mean(),
                                          df[i].std(),
                                          (df[i].std()/sqrt(len(df[i])))*t_val,
                                          df[i].mean() - (df[i].std() / sqrt(len(df[i]))) * t_val,
                                          df[i].mean() + (df[i].std() / sqrt(len(df[i]))) * t_val,
                                          df[i].min(), df[i].max())
            print (txt)
            with open('./Results/data_results.txt', 'a') as file:
                file.write(txt)
                file.write('\n')
                
# NOTE : ALSO ADD OTHER DATAFRAMES

with open('./Results/data_results.txt', 'a') as file:
    file.write('\n\n{:15} = {}'.format('sp', 'perceived spacing'))
    file.write('\n{:15} = {}'.format('own', 'perceived ownership'))
    file.write('\n{:15} = {}'.format('95% MoE','margin of error (one side of error bar) for 95% confidence interval'))
    file.write('\n{:15} = {}'.format('HR_diff', 'HR actual - HR reported'))
'''
 
 
 
 
 
 
 
 
 
 
 