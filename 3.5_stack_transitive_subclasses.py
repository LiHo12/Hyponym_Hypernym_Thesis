import pandas as pd
import os

### stack transitive subclasses rowwise
path_tsubclass = '/path/to/INTENSO/9_FINAL/data/matches/transitive-subclasses/'

files_tsubclass = os.listdir(path_tsubclass)

col_tsubclasses = ['_id', 'instance', 'class', 'frequency', 'pidspread', 'pldspread', 'modifications']

tsubclasses = pd.DataFrame(columns=col_tsubclasses) # initialize empty dataframe
for file in files_tsubclass:
    data = pd.read_csv(path_tsubclass+file, sep=";")
    del data['Unnamed: 0']
    #print(data.columns)
    print(file)

    tsubclasses = tsubclasses.append(data, ignore_index=True)

# remove duplicates
tsubclasses = tsubclasses.drop_duplicates(subset=['instance','class']) # drop duplicates
tsubclasses = tsubclasses.reset_index(drop=True) # reset indices

print('Found {} matches with webisa'.format(len(tsubclasses)))

# save file
tsubclasses.to_csv('/path/to/9_FINAL/data/matches/transitive-subclasses_matches_wo_duplicates.csv',sep=";")