# get modifications column since it contains the 'provids' (the id to join
# with the contexts)

import pandas as pd
import os
from helper import util_context

# read data and only get ids & modifications
data = pd.read_csv('/path/to/9_FINAL/data/machine_learning/two_class/distance/raw/all_distance_tuples.csv', sep=";")

# use length for debugging
train_check = data['id'].tolist()

# get tuples
tuples_files_folder = '/path/to/1_WebisALOD/tuplesdb_files/'

# debugging and loop through all modifications
all_matches = util_context.get_modifications(tuples_files_folder, train_check, '/path/to/9_FINAL/data/machine_learning/two_class/distance/appended/')

# Debugging
print('Found {} matches out of {}'.format(all_matches, len(data)))

