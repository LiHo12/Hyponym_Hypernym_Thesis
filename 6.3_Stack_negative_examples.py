import pandas as pd
import os

negative_examples_folder = '/path/to/9_FINAL/data/matches_pids/negative-examples/'
negative_examples = os.listdir(negative_examples_folder)

all_data = pd.DataFrame(columns=['_id', 'instance', 'class', 'frequency', 'pidspread', 'pldspread', 'modifications'])
counter = 0
complete_length = 0

checked_files = []
# loop through all files and stack them together
counter_2 = 0
for file in negative_examples:
    print(file)
    print(counter)
    print('-------')

    data = pd.read_csv(negative_examples_folder+file, sep=";")

    if len(data.columns) == 8:
        del data['Unnamed: 0']
        all_data = all_data.append(data, ignore_index=True)
        counter += 1
        complete_length += len(data)

print('Found {} negative examples'.format(complete_length))

all_data = all_data.drop_duplicates(subset=['instance', 'class'])  # drop duplicates
all_data = all_data.reset_index(drop=True)  # reset indices
all_data.to_csv('/path/to/9_FINAL/data/all_negative_examples.csv', sep=";")