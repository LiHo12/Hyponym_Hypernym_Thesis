import pandas as pd
import os

goldstandard = pd.read_csv('./webisalod_data/goldstandard_raw.csv', sep=";")
del goldstandard['Unnamed: 0']
id_goldstandard = goldstandard['id']
print(len(goldstandard))

# loop through all files in tuples
tuple_folder = '/path/to/tuplesdb_files/'
# download files from http://data.dws.informatik.uni-mannheim.de/webisadb/repo/tuplesdb.1.tar.gz
tuples_files = os.listdir(tuple_folder)

# checked files
#checked_files = os.listdir('path/to/merged/data')
# (file not in checked_files) and

# complete length
found_length = 0

for file in tuples_files:
    if ('#' not in file):
        print('Read file {}'.format(file))

        data = pd.read_csv(tuple_folder+file, sep=",")
        data = data[data['_id'].isin(id_goldstandard)]

        #all_data = pd.merge(data, goldstandard, left_on='_id', right_on='id')

        if len(data) > 0:
            print('Found match with {}'.format(file))
            #del all_data['_id']

            data.to_csv('/path/out'+file, sep=";") # see ./goldstandard_merged.csv file to see how a merge could look like

            found_length += len(data)

        print('--------------------------------------------------')

print('Found {} matches'.format(found_length))