import pandas as pd
import os
import ast

negative_examples_dir = '/path/to/9_FINAL/data/negative_examples/'
negative_examples_files = os.listdir(negative_examples_dir)

def extractPIDs(x):
    # get the pids in the array of dictionaries and append if not already existing
    pids = []

    for array in x:
        element = array['pids'].replace(';', '')

        pids.append(element)

    return pids

folderOut = '/path/to/9_FINAL/data/matches_pids/negative-examples/'

checked = os.listdir(folderOut)


for negative_example in negative_examples_files:
    if '~lock' not in negative_example:
        if negative_example not in checked:
            print('Load in {}'.format(negative_example))

            data = pd.read_csv(negative_examples_dir+negative_example, sep=";")
            del data['Unnamed: 0']

            data['modifications'] = data['modifications'].apply(ast.literal_eval) # convert into array
            data['modifications'] = data['modifications'].apply(lambda x: extractPIDs(x)) # get pids

            data.to_csv(folderOut+negative_example, sep=';') # export file