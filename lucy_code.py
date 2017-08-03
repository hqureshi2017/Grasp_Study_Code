import os
import pandas as pd
from math import sqrt

sub_numbers = list(range(1, 7))  ### !!! NEEDS TO CHANGE FOR REAL DATA
sub_id = []
gender = []
handedness = []
age = []

for sub_num in sub_numbers:
    # Read sub info file
    cur_sub = 'dummy_' + str(sub_num)  ### !!! NEEDS TO CHANGE FOR REAL DATA
    file_path = os.path.join('..','data', cur_sub, cur_sub + '.txt')
    with open(file_path) as f:
        data = f.readlines()
    # Get subject id
    cur_id = data[0].strip().split(':')[1].lstrip()
    sub_id.append(cur_id)
    # Get subject gender
    cur_gender = data[1].strip().split(':')[1].lstrip()
    gender.append(cur_gender)
    # Get subject handedness
    cur_handedness = data[2].strip().split(':')[1].lstrip()
    handedness.append(cur_handedness)
    # Get subject age
    cur_age = data[3].strip().split(':')[1]
    age.append(int(cur_age))

# Read in spreadsheet data for HR and questionnaire
file_path = os.path.join('..', 'data', 'secondary_measures' + '.xlsx')
HR_quest = pd.read_excel(file_path)

# Extact ownership and spacing data
video_own = []
video_sp = []
touch_own = []
touch_sp = []
audio_own = []
audio_sp = []
cur_block = []

for sub_num in sub_numbers:
    # Read sub info file
    cur_sub = 'dummy_' + str(sub_num)  ### !!! NEEDS TO CHANGE FOR REAL DATA
    path = os.path.join('..','data', cur_sub)
    file_path = path + '/' + cur_sub + '_data' + '.txt'
    with open(file_path) as f:
        data = f.readlines()

    for line in data:
        if line.strip().split(':')[0] == 'BLOCK':
            cur_block = line.strip().split(':')[1].lstrip()
        elif line.strip().split(':')[0] == 'TRIAL':
            pass
        elif line.strip().split(':')[0].split('_')[1] == 'SPACING':
            if cur_block == 'VIDEO':
                video_sp.append(-1 * int(line.strip().split(':')[1]))
            elif cur_block == 'TOUCH':
                touch_sp.append(-1 * int(line.strip().split(':')[1]))
            elif cur_block == 'AUDIO':
                audio_sp.append(-1 * int(line.strip().split(':')[1]))
        elif line.strip().split(':')[0].split('_')[1] == 'OWNERSHIP':
            if cur_block == 'VIDEO':
                video_own.append(int(line.strip().split(':')[1]))
            elif cur_block == 'TOUCH':
                touch_own.append(int(line.strip().split(':')[1]))
            elif cur_block == 'AUDIO':
                audio_own.append(int(line.strip().split(':')[1]))

data = pd.DataFrame({'age': age,
                     'gender': gender,
                     'handedness': handedness,
                     'audio_sp': audio_sp,
                     'audio_own': audio_own,
                     'video_sp': video_sp,
                     'video_own': video_own,
                     'touch_sp': touch_sp,
                     'touch_own': touch_own},
                    index=sub_id)

# Fix order of colums
col = data.columns
#          age    gender  handedness
new_col = [col[0], col[3], col[4],
#          video   audio   touch    SPACING
           col[8], col[2], col[6],
#         video   audio   touch    OWNERSHIP
           col[7], col[1], col[5]]
data = data[new_col]

# Convert gender and handedness to category
data['gender'] = data['gender'].astype('category')
data['handedness'] = data['handedness'].astype('category')

# Calculate difference scores
data['video_vs_audio_sp'] = data.video_sp - data.audio_sp
data['video_vs_touch_sp'] = data.video_sp - data.touch_sp
data['video_vs_audio_own'] = data.video_own - data.audio_own
data['video_vs_touch_own'] = data.video_own - data.touch_own

# Concatenate all data (HR, questionnaire, spacing, ownership)
data = pd.concat([data, HR_quest], axis=1)