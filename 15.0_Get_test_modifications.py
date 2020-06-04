# get modifications column since it contains the 'provids' (the id to join
# with the contexts)

import pandas as pd
import os
from helper import util_context

# read data and only get ids & modifications
data = pd.read_csv('/path/to/9_FINAL/data/machine_learning/two_class/count-based/test/count_test.csv', sep=";")

# use length for debugging
train_check = data['id'].tolist()
print(len(train_check))
# get tuples
tuples_files_folder = '/path/to/1_WebisALOD/tuplesdb_files/'

# debugging and loop through all modifications
all_matches = util_context.get_modifications(tuples_files_folder, train_check, '/path/to/9_FINAL/data/machine_learning/two_class/distance_test/modifications/')

# Debugging
print('Found {} matches out of {}'.format(all_matches, len(data)))

