import os
import pandas as pd

def get_modifications(tuplesfile_folder, check_ids, folder_out):
    """get modifications column since it contains the 'provids' (the id to join
     with the contexts)"""

    # get tuples
    tuples_files = os.listdir(tuplesfile_folder)

    # use length for debugging
    all_matches = 0

    # loop through all files
    for file in tuples_files:
        if '#' not in file:
            print('Start reading file {}'.format(file))
            tuples = pd.read_csv(tuplesfile_folder + file, sep=",")

            # reduce columns for efficiency
            tuples = tuples[['_id', 'instance', 'class', 'modifications']]

            # check if id is contained in train
            tuples = tuples[tuples['_id'].isin(check_ids)]

            # check if matches could be found
            if len(tuples) > 0:
                print('Found matches!')
                all_matches += len(tuples)
                # export csv
                tuples.to_csv(folder_out + file, sep=";")

            print('-------------------------------------------')

    return all_matches