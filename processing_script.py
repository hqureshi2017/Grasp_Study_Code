# Import scripts
import fx
import figures_data_check as fdc
import grasp_study_code as g
import figures_sp_measure_validation
import figures_2_sp_measure
import figures_3_own_measure
import figures_4_comparisons
import figures_5_loc_measure
import figures_6_calc_sp


# Import modules
fdc.modules(df)


# GET DATA INTO DATAFRAME
df = g.create(df)


# DATA CHECK
fdc.sp_own_data_check(df)
fdc.location_data_check(df)
fdc.validation_check(df)        


# VALIDATION SPACING MEASURE


# SPACING OUTCOME PLOT


# OWNERSHIP

    
# LOCATION OUTCOME PLOT (L & R)


# CALCULATED SPACING