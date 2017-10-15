# Note: The code below includes the figures as per the new plan (12/09/17).
# All other figures are in processing_script_old.py

# Import scripts
import grasp_study_code as g

import figures_2_sp_measure
import figures_2C_paired_comparisons
import sp_crossed_15cm
import figures_4_own_measure



# GET DATA INTO DATAFRAME
df = g.create()

g.text_demographic(df)



# SPACING OUTCOME PLOT
figures_2_sp_measure.sp_outcomes(df)                # 2A
figures_2_sp_measure.sp_diff_outcomes(df)           # 2B
figures_2C_paired_comparisons.paired(df)            # 2C


# LOCATION_15cm_CROSSED

sp_crossed_15cm.left_right_ng_g(df)                 # 3A + 3B


# OWNERSHIP
figures_4_own_measure.own_outcomes(df)              # 4A + 4B



