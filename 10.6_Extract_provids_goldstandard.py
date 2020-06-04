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

# get folder of goldstandard
folder = '/path/to/9_FINAL/data/goldstandard/'

# get data
data = pd.read_csv(folder + 'goldstandard_all.csv', sep=";", index_col=0)

# get provids
data['modifications'] = data['modifications'].apply(ast.literal_eval) # change datatype
data['provids'] = data['modifications'].apply(lambda x: extractProvids(x))

# delete unnecessary column
del data['modifications']

# export dataset
data.to_csv(folder+'goldstandard_with_provids.csv',sep=";", index=False)
