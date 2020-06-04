import pandas as pd
import os

# context
context_folder = '/path/to/1_WebisALOD/contextsdb_files/'
context_files = os.listdir(context_folder)

# all_data
all_data = pd.DataFrame()

# get ids of positives
#positives_id = ['12417510', '13078929', '22092812', '34709553', '22640311', '52253868', '148796396', '185415558', '191358866',
#                '196489438', '256949715', '257204518', '303678870', '316380244', '320305521', '359356386', '360377303']

# get negatives
negatives_id = ['181376908', '293624348', '239715226', '203211093',
                '57628507','409277661','235637522','49538950','860981',
                '303203756','430006305','57709577']

# loop through test and context data
for context in context_files:
    if (context != 'sentences.csv'):
        # get data
        data_context = pd.read_csv(context_folder+context,sep=",")

        # debug
        print('Start with {}'.format(context))
        print('-----------------------')

        # check with goldstandard
        data_context = data_context[data_context['provid'].isin(negatives_id)]

        # check if merges were found, export data
        if len(data_context) > 0:
            print(all_data)
            all_data = pd.concat([all_data, data_context])
            print(f'Found matches with {context}')
            print('-------------------------------')

# export data
all_data.to_csv('path/to/9_FINAL/data/goldstandard/goldstandard_negatives_extract_with_sentences_raw.csv',
                index=False, sep=";")