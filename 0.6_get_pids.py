import pandas as pd
import ast

all_data = pd.read_csv('./webisalod_data/goldstandard_all.csv', sep=";")
del all_data['Unnamed: 0']

def extractPIDs(x):
    # get the pids in the array of dictionaries and append if not already existing
    pids = []

    for array in x:
        element = array['pids'].replace(';', '')

        pids.append(element)

    return pids

all_data['modifications'] = all_data['modifications'].apply(ast.literal_eval)
all_data['modifications'] = all_data['modifications'].apply(lambda x: extractPIDs(x))

# print(len(data['modifications']))
all_data.to_csv('./webisalod_data/goldstandard_with_pids.csv', sep=";") # save data
