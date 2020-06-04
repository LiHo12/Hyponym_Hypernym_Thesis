import pandas as pd
import os

files_path = '/path/to/merged/data/'
checked_files = os.listdir(files_path)

all_data = pd.DataFrame()
complete_length = 0
for file in checked_files:

    data = pd.read_csv(files_path + file, sep=";")
    del data['Unnamed: 0']
    complete_length += len(data)
    all_data = pd.concat([all_data, data])

print(len(all_data))
print(complete_length)
all_data = all_data.reset_index(drop=True)
all_data.to_csv('./webisalod_data/goldstandard_all.csv', sep=";")