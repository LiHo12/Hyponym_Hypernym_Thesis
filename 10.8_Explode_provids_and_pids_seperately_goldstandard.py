import pandas as pd
import os


# explode entries test
folder = '/path/to/9_FINAL/data/goldstandard/'

# explode entries
data = pd.read_csv(folder+'goldstandard_exploded_provids_pids.csv', sep=";")

# explode data
exploded_data = pd.DataFrame(data.provid.str.split(';').tolist(), index=[data['id'], data['instance'], data['class'], data['pids']]).stack()
# print(exploded_data.shape)
exploded_data = exploded_data.reset_index([0, 'id', 'instance', 'class','pids'])  # var1 variable is currently labeled 0
exploded_data.columns = ['id', 'instance', 'class', 'pids', 'provid']
exploded_data = exploded_data[exploded_data.provid != ''] # delete empty observations
exploded_data = exploded_data.reset_index(drop=True)

# exploded 2
exploded_data = pd.DataFrame(exploded_data.pids.str.split(';').tolist(),
                             index=[exploded_data['id'], exploded_data['instance'], exploded_data['class'], exploded_data['provid']]).stack()
exploded_data = exploded_data.reset_index([0, 'id', 'instance', 'class', 'provid'])  # var1 variable is currently labeled 0
exploded_data.columns = ['id', 'instance', 'class', 'provid', 'pids']
exploded_data['provid'] = exploded_data.provid.apply(lambda x: str(x).replace(' ', ''))
exploded_data = exploded_data[exploded_data.pids != '']  # delete empty observations
exploded_data = exploded_data.reset_index(drop=True)

# export data
exploded_data.to_csv(folder+'goldstandard_exploded_separately.csv', sep=";", index=False)

print('------------------------------')

