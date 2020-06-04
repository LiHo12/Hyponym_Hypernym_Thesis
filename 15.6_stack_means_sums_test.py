import pandas as pd
import os


# stack all data together
all_data = pd.DataFrame()

# get folder of all files
folder = '/path/to/9_FINAL/data/machine_learning/two_class/distance_test/calculated_distance_'

all_data = pd.DataFrame()
checked_files = []

# loop through folders
for i in range(0,10):
    # specific folder
    specific_folder = folder + str(i) + '/'

    files = os.listdir(specific_folder)

    # loop through files and stack together
    for file in files:
        if file not in checked_files:
            data = pd.read_csv(specific_folder + file, sep=";", usecols=['id', 'pids', 'distance'])

            print(f'Start with file {file}')
            # stack to all data
            all_data = pd.concat([all_data, data],
                         ignore_index=True)

            checked_files.append(file)

        # group all data
all_data = all_data.groupby(by=['id', 'pids'])['distance'].agg(['sum', 'count']).reset_index()

# sums pivoted
sums = all_data[['id', 'pids', 'sum']]
# pivot table
sums = pd.pivot_table(sums, index='id', columns='pids', values='sum', aggfunc='first').fillna(0).reset_index()

# export data
sums.to_csv('/path/to/9_FINAL/data/machine_learning/two_class/distance_test/test_distance_sum.csv',
            sep=";", index=False)

# mean distance
all_data['mean_count'] = all_data['sum'] / all_data['count']
means = all_data[['id', 'pids', 'mean_count']]

# pivot table
means = pd.pivot_table(means, index='id', columns='pids', values='mean_count', aggfunc='first').fillna(0).reset_index()

means.to_csv('/path/to/9_FINAL/data/machine_learning/two_class/distance_test/test_distance_mean.csv',
            sep=";", index=False)