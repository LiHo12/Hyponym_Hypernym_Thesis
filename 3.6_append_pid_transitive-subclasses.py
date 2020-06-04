import pandas as pd
import ast

def extractPIDs(x):
    # get the pids in the array of dictionaries and append if not already existing
    pids = []

    for array in x:
        element = array['pids'].replace(';', '')

        pids.append(element)

    return pids

data = pd.read_csv('/path/to/9_FINAL/data/matches/transitive-subclasses_matches_wo_duplicates.csv',sep=";")
data['modifications'] = data['modifications'].apply(ast.literal_eval)
data['modifications'] = data['modifications'].apply(lambda x: extractPIDs(x))

print(len(data['modifications']))
data.to_csv('/path/to/9_FINAL/data/matches_pids/transitive-subclass_with_pid.csv', sep=";") # save data
