import os
import pandas as pd

folder = '/path/to/9_FINAL/data/machine_learning/two_class/sentence_length/sentence_length_'

# get ids of first fold
validation_fold = '/path/to/9_FINAL/data/machine_learning/two_class/one-hot-ecoding/train/cross_validation/validation/val_fold_0.csv' # TODO: change number
validation_data = pd.read_csv(validation_fold,
                              sep=";")

validation_data = validation_data.id.to_list()

# debug
# print(validation_data)
all_data = pd.DataFrame()

# loop through all folders
for i in range(1, 78):
    # get specific folder
    specific_folder = folder + str(i) + '/'

    # get files in that folder
    files = os.listdir(specific_folder)

    # loop through files to check for matches
    for file in files:
        data = pd.read_csv(specific_folder + file, sep=";", usecols=['id', 'provid', 'length_of_the_sentence'])

        # check if data has ids
        data = data[data.id.isin(validation_data)]

        # if length > 0, append to data
        if len(data) > 0:
            all_data = pd.concat([data, all_data])

    print(f'Finished with folder {str(i)}')

# export data
all_data.to_csv('/path/to/9_FINAL/data/machine_learning/two_class/sentence_length/fold_0/fold_0_all.csv', sep=";")