#from cumming_plot import cumming_plot
import pandas as pd
import numpy as np
from math import sqrt
import os
import random
from collections import OrderedDict
from scipy.stats import t

########### HASSAN ##############
#   Code contains the Dataframe for the Grasp Study Data.
#    Also has the calculations. Some of these were put on a new Dataframe due to bigger size
#    Code also contains analysis of results. 
################################## 

def create():
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

#%%            
# Dataframe for actual data
    d = OrderedDict({ 'age': age,
          'gender': gender,
          'handedness': handedness,
          'grasp_c15cm_sp': pd.Series(grasp_c15cm_sp, index = sub_id)*-1,
          'grasp_c15cm_own': pd.Series(grasp_c15cm_own, index = sub_id),
          'grasp_c15cm_r': pd.Series(grasp_c15cm_r, index = sub_id)-36,
          'grasp_c15cm_l': pd.Series(grasp_c15cm_l, index = sub_id)-36,
          'grasp_uc15cm_sp': pd.Series(grasp_uc15cm_sp, index = sub_id),
          'grasp_uc15cm_own': pd.Series(grasp_uc15cm_own, index = sub_id),
          'grasp_uc15cm_r': pd.Series(grasp_uc15cm_r, index = sub_id)-36,
          'grasp_uc15cm_l': pd.Series(grasp_uc15cm_l, index = sub_id)-36,
          'grasp_c24cm_sp': pd.Series(grasp_c24cm_sp, index = sub_id)*-1,
          'grasp_c24cm_own': pd.Series(grasp_c24cm_own, index = sub_id),
          'grasp_c24cm_r': pd.Series(grasp_c24cm_r, index = sub_id)-36,
          'grasp_c24cm_l': pd.Series(grasp_c24cm_l, index = sub_id)-36,
          'grasp_uc24cm_sp': pd.Series(grasp_uc24cm_sp, index = sub_id),
          'grasp_uc24cm_own': pd.Series(grasp_uc24cm_own, index = sub_id),
          'grasp_uc24cm_r': pd.Series(grasp_uc24cm_r, index = sub_id)-36,
          'grasp_uc24cm_l': pd.Series(grasp_uc24cm_l, index = sub_id)-36,
          'nograsp_c24cm_sp': pd.Series(nograsp_c24cm_sp, index = sub_id)*-1,
          'nograsp_c24cm_r': pd.Series(nograsp_c24cm_r, index = sub_id)-36,
          'nograsp_c24cm_l': pd.Series(nograsp_c24cm_l, index = sub_id)-36,
          'nograsp_c15cm_sp': pd.Series(nograsp_c15cm_sp, index = sub_id)*-1,
          'nograsp_c15cm_r': pd.Series(nograsp_c15cm_r, index = sub_id)-36,
          'nograsp_c15cm_l': pd.Series(nograsp_c15cm_l, index = sub_id)-36,
          'nograsp_uc15cm_sp': pd.Series(nograsp_uc15cm_sp, index = sub_id),
          'nograsp_uc15cm_r': pd.Series(nograsp_uc15cm_r, index = sub_id)-36,
          'nograsp_uc15cm_l': pd.Series(nograsp_uc15cm_l, index = sub_id)-36,
          'nograsp_uc24cm_sp': pd.Series(nograsp_uc24cm_sp, index = sub_id),
          'nograsp_uc24cm_r': pd.Series(nograsp_uc24cm_r, index = sub_id)-36,
          'nograsp_uc24cm_l': pd.Series(nograsp_uc24cm_l, index = sub_id)-36,
            'L24_1':pd.Series(val_L24cm_val1, index = sub_id),
            'L24_2':pd.Series(val_L24cm_val2, index = sub_id),
            'L15_1':pd.Series(val_L15cm_val1, index = sub_id),
            'L15_2':pd.Series(val_L15cm_val2, index = sub_id),
            'R15_1':pd.Series(val_R15cm_val1, index = sub_id)*-1,
            'R15_2':pd.Series(val_R15cm_val2, index = sub_id)*-1,
            'R24_1':pd.Series(val_R24cm_val1, index = sub_id)*-1,
            'R24_2':pd.Series(val_R24cm_val2, index = sub_id)*-1,

            'loc_R_L24_1':pd.Series(val_loc_R_L24cm_val1, index = sub_id)-36,
            'loc_R_L24_2':pd.Series(val_loc_R_L24cm_val2, index = sub_id)-36,
            'loc_R_L15_1':pd.Series(val_loc_R_L15cm_val1, index = sub_id)-36,
            'loc_R_L15_2':pd.Series(val_loc_R_L15cm_val2, index = sub_id)-36,
            'loc_R_R15_1':pd.Series(val_loc_R_R15cm_val1, index = sub_id)-36,
            'loc_R_R15_2':pd.Series(val_loc_R_R15cm_val2, index = sub_id)-36,
            'loc_R_R24_1':pd.Series(val_loc_R_R24cm_val1, index = sub_id)-36,
            'loc_R_R24_2':pd.Series(val_loc_R_R24cm_val2, index = sub_id)-36,
            
            'loc_L_L24_1':pd.Series(val_loc_L_L24cm_val2, index = sub_id)-36,
            'loc_L_L24_2':pd.Series(val_loc_L_L24cm_val1, index = sub_id)-36,
            'loc_L_L15_1':pd.Series(val_loc_L_L15cm_val2, index = sub_id)-36,
            'loc_L_L15_2':pd.Series(val_loc_L_L15cm_val1, index = sub_id)-36,
            'loc_L_R15_1':pd.Series(val_loc_L_R15cm_val2, index = sub_id)-36,
            'loc_L_R15_2':pd.Series(val_loc_L_R15cm_val1, index = sub_id)-36,
            'loc_L_R24_1':pd.Series(val_loc_L_R24cm_val2, index = sub_id)-36,
            'loc_L_R24_2':pd.Series(val_loc_L_R24cm_val1, index = sub_id)-36})
    
    # Create dataframe
    df = pd.DataFrame(d, index = sub_id) 
    
    # Convert gender and handedness to category
    df['gender'] = df['gender'].astype('category')
    df['handedness'] = df['handedness'].astype('category')
    
    # Calculate spacing difference scores
    df['24cm_grasp_vs_nograsp_sp'] =  df.nograsp_uc24cm_sp - df.grasp_uc24cm_sp
    df['24cm_grasp_vs_nograsp_r'] = ((df.nograsp_uc24cm_r) - (df.grasp_uc24cm_r))*-1
    df['24cm_grasp_vs_nograsp_l'] = ((df.nograsp_uc24cm_l) - (df.grasp_uc24cm_l))*-1
    
    df['15cm_grasp_vs_nograsp_sp'] = df.nograsp_uc15cm_sp - df.grasp_uc15cm_sp
    df['15cm_grasp_vs_nograsp_r'] = ((df.nograsp_uc15cm_r) - (df.grasp_uc15cm_r))*-1
    df['15cm_grasp_vs_nograsp_l'] = ((df.nograsp_uc15cm_l) - (df.grasp_uc15cm_l))*-1
    
    df['-15cm_grasp_vs_nograsp_sp'] = df.nograsp_c15cm_sp - df.grasp_c15cm_sp
    df['-15cm_grasp_vs_nograsp_r'] = ((df.nograsp_c15cm_r) - (df.grasp_c15cm_r))*-1
    df['-15cm_grasp_vs_nograsp_l'] = ((df.nograsp_c15cm_l) - (df.grasp_c15cm_l))*-1
    
    df['-24cm_grasp_vs_nograsp_sp'] = df.nograsp_c24cm_sp - df.grasp_c24cm_sp
    df['-24cm_grasp_vs_nograsp_r'] = ((df.nograsp_c24cm_r) - (df.grasp_c24cm_r))*-1
    df['-24cm_grasp_vs_nograsp_l'] = ((df.nograsp_c24cm_l) - (df.grasp_c24cm_l))*-1
    
    # Effect of Distance + Midline on spacing
    df['-15_vs_-24_sp'] = df['-15cm_grasp_vs_nograsp_sp'] - df['-24cm_grasp_vs_nograsp_sp']
    df['15_vs_24_sp'] = df['15cm_grasp_vs_nograsp_sp'] - df['24cm_grasp_vs_nograsp_sp']
    df['-15_vs_15_sp'] = df['-15cm_grasp_vs_nograsp_sp'] - df['15cm_grasp_vs_nograsp_sp']
    df['-24_vs_24_sp'] = df['-24cm_grasp_vs_nograsp_sp'] - df['24cm_grasp_vs_nograsp_sp']
    
    # Effect of Distance + Midline on ownership
    df['-15_vs_-24_own'] = df.grasp_c15cm_own - df.grasp_c24cm_own
    df['15_vs_24_own'] = df.grasp_uc15cm_own - df.grasp_uc24cm_own
    df['-15_vs_15_own'] = df.grasp_c15cm_own - df.grasp_uc15cm_own
    df['-24_vs_24_own'] = df.grasp_c24cm_own - df.grasp_uc24cm_own
    
    # Calculation Spacing
    df['-24_calc_spacing_Grasp'] = (df.grasp_c24cm_l) - (df.grasp_c24cm_r)
    df['-15_calc_spacing_Grasp'] = (df.grasp_c15cm_l) - (df.grasp_c15cm_r)
    df['15_calc_spacing_Grasp'] = ((df.grasp_uc15cm_l) - (df.grasp_uc15cm_r))*-1
    df['24_calc_spacing_Grasp'] = ((df.grasp_uc24cm_l) - (df.grasp_uc24cm_r))*-1
    
    df['-24_calc_spacing_No_Grasp'] = (df.nograsp_c24cm_l) - (df.nograsp_c24cm_r)
    df['-15_calc_spacing_No_Grasp'] = (df.nograsp_c15cm_l) - (df.nograsp_c15cm_r)
    df['15_calc_spacing_No_Grasp'] = ((df.nograsp_uc15cm_l) - (df.nograsp_uc15cm_r))*-1
    df['24_calc_spacing_No_Grasp'] = ((df.nograsp_uc24cm_l) - (df.nograsp_uc24cm_r))*-1
    
    df['-24_calc_sp_diff'] = df['-24_calc_spacing_No_Grasp'] - df['-24_calc_spacing_Grasp']
    df['-15_calc_sp_diff'] = df['-15_calc_spacing_No_Grasp'] - df['-15_calc_spacing_Grasp']
    df['15_calc_sp_diff'] = df['15_calc_spacing_No_Grasp'] - df['15_calc_spacing_Grasp']
    df['24_calc_sp_diff'] = df['24_calc_spacing_No_Grasp'] - df['24_calc_spacing_Grasp']
    
    df['-15cm_calc_vs_per_sp_diff'] = df['-15_calc_sp_diff'] - df['-15cm_grasp_vs_nograsp_sp']
    
    # Calculated Spacing_VALIDATION
    df['sp_calc_c24_1'] = (df.loc_L_R24_1) - (df.loc_R_R24_1)
    df['sp_calc_c24_2'] = (df.loc_L_R24_2) - (df.loc_R_R24_2)
    df['sp_calc_c15_1'] = (df.loc_L_R15_1) - (df.loc_R_R15_1)
    df['sp_calc_c15_2'] = (df.loc_L_R15_2) - (df.loc_R_R15_2)
    df['sp_calc_uc24_1'] = ((df.loc_L_L24_1) - (df.loc_R_L24_1))*-1
    df['sp_calc_uc24_2'] = ((df.loc_L_L24_2) - (df.loc_R_L24_2))*-1
    df['sp_calc_uc15_1'] = ((df.loc_L_L15_1) - (df.loc_R_L15_1))*-1
    df['sp_calc_uc15_2'] = ((df.loc_L_L15_2) - (df.loc_R_L15_2))*-1
 
    # Spacing Measure Difference_VALIDATION
    df['val_-24_sp_diff_1'] = df['sp_calc_c24_1'] - df['R24_1']
    df['val_-24_sp_diff_2'] = df['sp_calc_c24_2'] - df['R24_2']
    df['val_-15_sp_diff_1'] = df['sp_calc_c15_1'] - df['R15_1']
    df['val_-15_sp_diff_2'] = df['sp_calc_c15_2'] - df['R15_2']
    df['val_24_sp_diff_1'] = df['sp_calc_uc24_1'] - df['L24_1']
    df['val_24_sp_diff_2'] = df['sp_calc_uc24_2'] - df['L24_2']
    df['val_15_sp_diff_1'] = df['sp_calc_uc15_1'] - df['L15_1']
    df['val_15_sp_diff_2'] = df['sp_calc_uc15_2'] - df['L15_2']
    
    df.loc['sub01','nograsp_c15cm_r'] = np.nan
    df.loc['sub01','nograsp_c15cm_l'] = np.nan
    df.loc['sub01','grasp_c15cm_r'] = np.nan
    df.loc['sub01','grasp_c15cm_l'] = np.nan
    df.loc['sub01','-15cm_grasp_vs_nograsp_r'] = np.nan 
    df.loc['sub01','-15cm_grasp_vs_nograsp_l'] = np.nan

    return df

# Way to inspect dataframe
       # print(df.to_string())

#%%

def calc(df):
        
        
    calc_sp = [df['-24_calc_spacing_Grasp'].values] + [df['-15_calc_spacing_Grasp'].values] + [df['15_calc_spacing_Grasp'].values] + [df['24_calc_spacing_Grasp'].values]+[df['-24_calc_spacing_No_Grasp'].values] + [df['-15_calc_spacing_No_Grasp'].values] + [df['15_calc_spacing_No_Grasp'].values] + [df['24_calc_spacing_No_Grasp'].values]
    perc_sp = [df['grasp_c24cm_sp'].values] + [df['grasp_c15cm_sp'].values]+[df['grasp_uc15cm_sp'].values] + [df['grasp_uc24cm_sp'].values]+[df['nograsp_c24cm_sp'].values] + [df['nograsp_c15cm_sp'].values] + [df['nograsp_uc15cm_sp'].values]  + [df['nograsp_uc24cm_sp'].values]
   
    
#    new_val = pd.DataFrame()
#    new_val['perceived'] = [df1['sp_perceived_c24'].values]+[df1['sp_perceived_c15'].values]+[df1['sp_perceived_uc24'].values]+[df1['sp_perceived_uc15'].values]
#    new_val['calculated']= [df1['sp_calc_c24_total'].values]+ [df1['sp_calc_c15_total'].values]+[df1['sp_calc_uc24_total'].values] +  [df1['sp_calc_uc15_total'].values]
    
    return df
#%%
  
####################################################
# CALCULATE DIFFERENCE VALUES AND SUMMARY STATISTICS
####################################################

#!!! NOTE _ NEED TO EDIT THE BELOW CODE

def text_demographic(df):
    # Loop through items in dataframe and calculate basic summary statistics
    txt = ['{:<15} {}'.format('Male', sum(df.gender == 'M')),
           '{:<15} {}'.format('Female', sum(df.gender == 'F')),
           '{:<15} {}'.format('Right handed', sum(df.handedness == 'R')),
           '{:<15} {}'.format('Left handed', sum(df.handedness == 'L')),
           '{:<15} mean = {:>4.1f}   SD = {:>3.1f}   min = {:>2.0f}   '
           'max = {:>2.0f}'.format('age', df['age'].mean(),
                                   df['age'].std(), df['age'].min(), df['age'].max())]
    
    for loop, line in enumerate(txt):
        if loop == 0:
            flag = 'w'
        else:
            flag = 'a'
        with open('./Results/data_results.txt', flag) as file:
            file.write(line)
            file.write('\n')
    
    with open('./Results/data_results.txt', flag) as file:            
        file.write('\n\n{:8} = {}'.format('sp', 'perceived spacing'))
        file.write('\n{:8} = {}'.format('own', 'perceived ownership'))
        file.write('\n{:8} = {}'.format('95% MoE','margin of error (one side of error bar) for 95% confidence interval'))
        file.write('\n{:8} = {}'.format('l','perceived location of LEFT hand'))
        file.write('\n{:8} = {}'.format('r','perceived location of RIGHT hand'))
        file.write('\n{:8} = {}'.format('uc','hands are uncrossed'))
        file.write('\n{:8} = {}'.format('c','hands are crossed'))    
            


def text_figure(fig_name, df_col_names, df):
    
    with open('./Results/data_results.txt', 'a') as file:
        file.write('\n\n')
        file.write('=' * len(fig_name))
        file.write('\n')
        file.write(fig_name)
        file.write('\n')
        file.write('=' * len(fig_name))
        file.write('\n\n')
    
    # Loop through items in dataframe and calculate basic summary statistics
    t_val = float(t.ppf([0.975], 30))
    txt = '{:<30s} {:<8s} {:<8s} {:<8s} {:<10s} {:^16s} {:<6s} {:<6s}'.format(
          'measure', 'count', 'mean', 'SD', '95% MoE', '95%CI', ' min', ' max')
    dash = len(txt) * '-'
    with open('./Results/data_results.txt', 'a') as file:
        file.write(dash)
        file.write('\n')
        file.write(txt)
        file.write('\n')
        file.write(dash)
        file.write('\n')
         
    for col_name in df_col_names:
        txt = '{:<30s} {:<8d} {:<8.2f} {:<8.2f} {:<10.2f} {:^8.2f} {:<8.2f} {:<6.2f} {:<6.2f}'.format(
              col_name,
              df[col_name].count(),
              df[col_name].mean(),
              df[col_name].std(),
              (df[col_name].std()/sqrt(len(df[col_name])))*t_val,
              df[col_name].mean() - (df[col_name].std() / sqrt(len(df[col_name]))) * t_val,
              df[col_name].mean() + (df[col_name].std() / sqrt(len(df[col_name]))) * t_val,          
              df[col_name].min(), df[col_name].max())
        with open('./Results/data_results.txt', 'a') as file:
            file.write(txt)
            file.write('\n')
