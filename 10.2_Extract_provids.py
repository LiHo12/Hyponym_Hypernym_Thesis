import pandas as pd
import ast
import os

# extract provids
def extractProvids(x):
    # get the Provids in the array of dictionaries and append if not already existing
    provids = []

    for array in x:
        element = array['provids'].replace('\'', ',')
        element += 'patterns: ' + array['pids'].replace('\'', ',')
        provids.append(element)

    return provids

# test
# get appended files
fold_folder = '/path/to/9_FINAL/data/machine_learning/two_class/distance/appended/'
files_all = os.listdir(fold_folder)

folder_out = '/path/to/9_FINAL/data/machine_learning/two_class/distance/modifications/'

for file in files_all:
    data = pd.read_csv(fold_folder+file, sep=";")
    del data['Unnamed: 0']
    # debug
    #print(test.head())
    if len(data) > 0:
        data['modifications'] = data['modifications'].apply(ast.literal_eval) # change datatype
        data['provids'] = data['modifications'].apply(lambda x: extractProvids(x))
        #test['pids'] = test['modifications'].apply(lambda x: extractPids(x))

        del data['modifications']

        # export dataset
        data.to_csv(folder_out+file,sep=";")

    #print(test.head())
