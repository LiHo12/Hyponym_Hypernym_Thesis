import pandas as pd
import os

# context
context_folder = '/path/to/1_WebisALOD/contextsdb_files/'
context_files = os.listdir(context_folder)

# exploded pids and provids folder
folder = '/path/to/9_FINAL/data/goldstandard/'
folder_out = folder + 'contexts/'

# get data of goldstandard
goldstandard = pd.read_csv(folder + 'goldstandard_exploded_separately.csv', sep=";")

# loop through test and context data
for context in context_files:
    if (context != 'sentences.csv'):
        # get data
        data_context = pd.read_csv(context_folder+context,sep=",")

        # debug
        print('Start with {}'.format(context))
        print('-----------------------')

        # check with goldstandard

        merged_data = pd.merge(data_context, goldstandard, left_on='provid', right_on='provid')

        # check if merges were found, export data
        if len(merged_data) > 0:
            del merged_data['_id'] # unnecessary sentence id
            del merged_data['pld'] # unnecessary pld

            # lower case sentence column
            merged_data['sentence'] = merged_data['sentence'].str.lower()

            merged_data.to_csv(folder_out+context,sep=";", index=False)
            print(f'Found matches with {context}')
            print('-------------------------------')
