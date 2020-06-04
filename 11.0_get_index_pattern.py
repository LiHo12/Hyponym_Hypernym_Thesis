import pandas as pd

# Note that some instances can be placed after the class.
# Thus, identify the patterns where the instance is after the class.
# For patterns, where the index of the instance is before class, denote column with 1.
# Otherwise, denote column with -1.

data = pd.read_csv('/path/to/9_FINAL/data/context/pattern_details.csv', sep=",")

# print(data.head()) # debugging


def get_position_of_instance_and_class(x):
    pattern_array_split = x['pattern'].replace(',','').split(' ')
    print(pattern_array_split)
    # get index of instance
    index_instance = pattern_array_split.index('npi')

    # get index of class
    index_class = pattern_array_split.index('npc')

    if index_class > index_instance:
        # class is after instance
        return 1
    else:
        return -1

# lower casing string
data['pattern'] = data['pattern'].str.lower()
data['position'] = data.apply(lambda x: get_position_of_instance_and_class(x), axis=1)


# save file
data.to_csv('/path/to/9_FINAL/data/context/pattern_details_position.csv', sep=";")