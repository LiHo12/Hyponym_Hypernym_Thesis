import pandas as pd
import ast
import os

def extractPIDs(x):
    # get the pids in the array of dictionaries and append if not already existing
    pids = []

    for array in x:
        element = array['pids'].replace(';', '')

        pids.append(element)

    return pids

folderIn = '/path/to/1_WebisALOD/tuplesdb_files/'
files = os.listdir(folderIn)

checked = '/path/to/9_FINAL/data/pids/'
files_checked = os.listdir(checked)
files_checked = [x.replace('pids_', '') for x in files_checked]

folderOut = '/media/linda/INTENSO/9_FINAL/data/pids/'
for file in files:
    if file not in files_checked:
         print('Start parsing file {}'.format(file))

         pids = pd.read_csv(folderIn + file, sep=",")

         if '_id' not in pids.columns:
             pids = pd.read_csv(folderIn + file, sep=",", header=None)
             pids.columns = ['_id','instance','class','frequency','pidspread','pldspread','modifications']

         pids = pids[['_id', 'modifications']] # subset to relevant columns

         pids['modifications'] = pids['modifications'].apply(ast.literal_eval) # convert into array
         pids['modifications'] = pids['modifications'].apply(lambda x: extractPIDs(x)) # get pids

         # rename columns
         pids.columns = ['id', 'pids']
         # print(pids.head(35)) # SANITY CHECK
         # save dataframe

         # write file out
         pids.to_csv(folderOut+'pids_'+file, sep=";")

         print('Finished parsing file {}'.format(file))
         print('------------------------------------------------')


