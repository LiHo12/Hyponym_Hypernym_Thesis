import pandas as pd
import numpy as np
import os
from pathlib import Path
folder = '/path/to/9_FINAL/data/machine_learning/two_class/distance/calculated_distance_'

folder_out = '/path/to/9_FINAL/data/machine_learning/two_class/distance/grouping/Run_2/validation_'
validation_folder ='/path/to/9_FINAL/data/machine_learning/two_class/one-hot-ecoding/train/cross_validation/validation/val_fold_'

# loop through all validations to get all validations sets
for i in range(1, 5):
    print(f'Start with validation fold {str(i)}')
    validation_ids = validation_folder + str(i) + '.csv'

    # get ids of validation file
    validation_ids = pd.read_csv(validation_ids, sep=';')
    validation_ids = validation_ids['id'].to_list()

    # debug
    # print(len(validation_ids))

    # get path to folderout
    specific_folder_out = folder_out + str(i+1) + '/'
    Path(specific_folder_out).mkdir(parents=True, exist_ok=True)

    # loop through calculated distances and check whether the data exists
    for k in range(1,79): # TODO: fill number of folders
        # get all files from that folder
        specific_folder_in = folder + str(k) + '/'

        specific_folder_out_out = specific_folder_out + 'under_folder_' + str(k) + '/'
        files_in_specific_folder_in = os.listdir(specific_folder_in)

        # make directory if not exists
        Path(specific_folder_out_out).mkdir(parents=True, exist_ok=True)
        for file in files_in_specific_folder_in:
                # read data
            data = pd.read_csv(specific_folder_in+file, sep=";", index_col=0)
            # subset data to relevant columns
            data = data[['id', 'pids', 'distance']]

            # check whether data is contained in id of validation data
            data = data[data.id.isin(validation_ids)].reset_index(drop=True)

            # if length is bigger than zero, concat to original data
            if len(data) > 0:
                # export data
                data.to_csv(specific_folder_out_out+file, sep=";")
                # debugging
                print(f'Found matches with {file}!')

        print('----------------------------------------------------------')
    print('__________________________________________________-')


