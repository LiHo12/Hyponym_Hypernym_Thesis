import pandas as pd
import os

# context
context_folder = '/path/to/INTENSO/1_WebisALOD/contextsdb_files/'
context_files = os.listdir(context_folder)


# exploded pids and provids folder
folder_in_name = '/path/in/'
folder_in_files = os.listdir(folder_in_name)

# checked files
checked_files = os.listdir('path/to/checked/folder/')

checked_sentences = [x.split('_')[1] for x in checked_files]

# loop through test and context data
for context in context_files:
    data_context = pd.read_csv(context_folder+context,sep=",")

    if (context != 'sentences.csv') and (context not in checked_sentences):
        # print(data_context.head())
        print('Start with {}'.format(context))
        print('-----------------------')

        # loop through test files and merge data
        for file in folder_in_files:
            # check if already in test
            check_test = file.replace('.csv', '')
            check_test = check_test + '_' + context


            data = pd.read_csv(folder_in_name+file, sep=";")
            del data['Unnamed: 0']

                # print(data.head())
            merged_data = pd.merge(data_context, data, left_on='provid', right_on='provid')

                # check if merges were found, export data
            if len(merged_data) > 0:
                filename = file.replace('.csv', '')

                export_file = filename + '_' + context
                del merged_data['_id'] # unnecessary sentence id
                del merged_data['pld'] # unnecessary pld

                    # lower case sentence column
                merged_data['sentence'] = merged_data['sentence'].str.lower()

                merged_data.to_csv('/path/out'+export_file,sep=";")
                print('Found matches with {}'.format(filename))
