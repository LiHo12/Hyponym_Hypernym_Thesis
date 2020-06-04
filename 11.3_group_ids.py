import os
import pandas as pd

check_files = '/path/to/9_FINAL/data/machine_learning/two_class/distance/grouping/Run_2/validation_1/under_folder_'

# loop through all folders to stack data
for i in range(63, 79):

    # initialize all data
    all_data = pd.DataFrame(columns=['id', 'pids', 'sum', 'count'])

    # get right index for folder
    check_folder = check_files + str(i) + '/'

    # get all the files in the folder
    files = os.listdir(check_folder)
    print(f'Folder {str(i)}')

    #for file in files:
    for file in files:
        data = pd.read_csv(check_folder+file, sep=";", index_col=0)

        # sanity check
        #data['key'] = data['pids'] + data['id'].astype(str)
        #print(data.key.value_counts())

        # aggregate sum and count
        data = data.groupby(by=['id','pids'])['distance'].agg(['sum','count']).reset_index()

        # add to whole data set
        all_data = data.set_index(['id','pids']).add(all_data.set_index(['id','pids']), fill_value=0).reset_index()
        # sanity check
        # print(all_data[all_data['count'] > 1])

    # Save intermediate result
    all_data.to_csv('/path/to/9_FINAL/data/machine_learning/two_class/distance/grouping/Run_3/validation_fold_0/count_sum_' + str(i) + '.csv', sep=";", index=False)

    # sanity check
    print(f'Finished with folder {str(i)}')
    print('--------------------------------')