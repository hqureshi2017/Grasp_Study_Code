import pandas as pd
import os
from collections import OrderedDict

def create_df():
        
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
            
            'loc_L_L24_1':val_loc_L_L24cm_val1,
            'loc_L_L24_2':val_loc_L_L24cm_val2,
            'loc_L_L15_1':val_loc_L_L15cm_val1,
            'loc_L_L15_2':val_loc_L_L15cm_val2,
            'loc_L_R15_1':val_loc_L_R15cm_val1,
            'loc_L_R15_2':val_loc_L_R15cm_val2,
            'loc_L_R24_1':val_loc_L_R24cm_val1,
            'loc_L_R24_2':val_loc_L_R24cm_val2})
    
    df = pd.DataFrame(d, index = sub_id)
    return df



def _jitter(data, jit=None):
    if jit is None:
        jit = 0.005

    a = data[0]
    b = data[1]

    duplicate_a = []
    duplicate_b = []
    jitter_a = [''] * len(a)
    jitter_b = [''] * len(b)
    jitter_a[0] = 0
    jitter_b[0] = 0

    for i in np.arange(1, len(a), 1):
        a_val = a.values[i]
        b_val = b.values[i]
        if a_val in a.values[0:i]:
            duplicate_a.append(a_val)
            val = jit * duplicate_a.count(a_val)
            jitter_a[i] = val
        else:
            jitter_a[i] = 0

        if b_val in b.values[0:i]:
            duplicate_b.append(b_val)
            val = jit * duplicate_b.count(b_val)
            jitter_b[i] = val
        else:
            jitter_b[i] = 0
    return jitter_a, jitter_b
    