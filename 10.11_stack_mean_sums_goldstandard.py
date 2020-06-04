import pandas as pd
import os


# stack all data together
all_data = pd.DataFrame()

# get folder of all files
folder = '/path/to/9_FINAL/data/goldstandard/contexts_calculated/'

files_in = os.listdir(folder)

# loop through all files to stack the data together
for file in files_in:
    print(f'Start with file {file}')
    # load data
    data = pd.read_csv(folder+file, sep=";",
                       usecols=['id', 'pids', 'distance'])

    # stack to all data
    all_data = pd.concat([all_data, data],
                         ignore_index=True)

# group all data
all_data = all_data.groupby(by=['id', 'pids'])['distance'].agg(['sum', 'count']).reset_index()

# sums pivoted
sums = all_data[['id', 'pids', 'sum']]
# pivot table
sums = pd.pivot_table(sums, index='id', columns='pids', values='sum', aggfunc='first').fillna(0).reset_index()

# get labels
labels = pd.read_csv('/path/to/9_FINAL/data/goldstandard/goldstandard_raw.csv',
                     sep=";", index_col=0)
labels.columns = ['id', 'label']

# merge labels to sums
sums = pd.merge(sums, labels, how='left',
                right_on='id', left_on='id')
print(sums.shape)

# export data
sums.to_csv('/path/to/9_FINAL/data/goldstandard/goldstandard_distance_sum.csv',
            sep=";", index=False)

# mean distance
all_data['mean_count'] = all_data['sum'] / all_data['count']
means = all_data[['id', 'pids', 'mean_count']]

# pivot table
means = pd.pivot_table(means, index='id', columns='pids', values='mean_count', aggfunc='first').fillna(0).reset_index()

# merge labels to means
means = pd.merge(means, labels, how='left',
                right_on='id', left_on='id')
print(means.shape)

means.to_csv('/path/to/9_FINAL/data/goldstandard/goldstandard_distance_mean.csv',
            sep=";", index=False)