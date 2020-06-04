import pandas as pd
import os


# explode entries test
folder = '/path/to/9_FINAL/data/goldstandard/'

# explode entries
data = pd.read_csv(folder + 'goldstandard_with_provids.csv', sep=";")

# replace unnecessary characters
data['provids'] = [x.replace('\'', '') for x in data['provids']]
data['provids'] = [x.replace('[', '') for x in data['provids']]
data['provids'] = [x.replace(']', '') for x in data['provids']]
data['provids'] = [x.replace(' ', '') for x in data['provids']]


# explode data
exploded_data = pd.DataFrame(data.provids.str.split(',').tolist(), index=[data['_id'], data['instance'], data['class']]).stack()
exploded_data = exploded_data.reset_index([0, '_id', 'instance', 'class'])  # var1 variable is currently labeled 0
exploded_data.columns = ['id', 'instance', 'class', 'provid']
exploded_data = exploded_data[exploded_data.provid != ''] # delete empty observations
exploded_data = exploded_data.reset_index(drop=True)
# print(exploded_data.head())

# get pid column and provid
exploded_data['pids'] = exploded_data.provid.apply(lambda x: str(x).split(';patterns:')[1])
exploded_data['provid'] = exploded_data.provid.apply(lambda x: str(x).split('patterns:')[0])

# export data
exploded_data.to_csv(folder+'goldstandard_exploded_provids_pids.csv', sep=";", index=False)

print('------------------------------')
