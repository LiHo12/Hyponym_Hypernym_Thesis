import pandas as pd
import os

# read subclasses into memory
subclasses = pd.read_csv('/path/to/9_FINAL/data/matches/subclass_matches_wo_duplicates.csv', sep=";")
del subclasses['Unnamed: 0']
print(len(subclasses)) # print length

# get all pids
pids_folder = '/path/to/9_FINAL/data/pids/'
pids = os.listdir(pids_folder)

# merge for all existing files the right pids
for file in pids:
    pid = pd.read_csv(pids_folder+file, sep=";")
    del pid['Unnamed: 0']

    all_data = pd.merge(subclasses, pid, right_on='id', left_on='_id')
    del all_data['_id'] # delete column since it is a duplicate

    all_data.to_csv('/path/to/9_FINAL/data/matches_pids/subclass/subclass_' + file, sep=";")

merged_pids_folder = '/path/to/9_FINAL/data/matches_pids/subclass/'
merged_pids = os.listdir(merged_pids_folder)

all_merged_pids = pd.DataFrame()

# concat the data back
for file in merged_pids:
    #print(file)
    data = pd.read_csv(merged_pids_folder+file, sep=";")
    del data['Unnamed: 0']

    all_merged_pids = all_merged_pids.append(data, ignore_index=True)

all_merged_pids.to_csv('/path/to/9_FINAL/data/matches_pids/subclass_with_pid.csv', sep=";")
print(len(all_merged_pids))