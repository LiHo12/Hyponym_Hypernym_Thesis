import os
import pandas as pd

### transitive-types
path_ttypes = '/path/to/9_FINAL/data/matches/transitive-types/'

files_ttypes = os.listdir(path_ttypes)

col_ttypes = ['instance', 'class', '_id', 'frequency', 'pidspread', 'pldspread']

ttypes = pd.DataFrame(columns=col_ttypes) # initialize empty dataframe
for file in files_ttypes:
    data = pd.read_csv(path_ttypes+file, sep=";")

    ttypes = ttypes.append(data[col_ttypes], ignore_index=True)

print('{} matches for transitive-types'.format(len(ttypes)))

ttypes.to_csv('/path/to/9_FINAL/data/matches/transitive-types_matches.csv', sep=";")

### types
path_types = '/path/to/9_FINAL/data/matches/types/'

files_types = os.listdir(path_types)

col_types = ['instance', 'class', '_id', 'frequency', 'pidspread', 'pldspread']

types = pd.DataFrame(columns=col_types) # initialize empty dataframe
for file in files_types:
    data = pd.read_csv(path_types+file, sep=";")

    types = types.append(data[col_types], ignore_index=True)

print('{} matches for types'.format(len(types)))

types.to_csv('/path/to/9_FINAL/data/matches/types_matches.csv', sep=";")

### subclass
path_subclass = '/path/to/9_FINAL/data/matches/subclass/'

files_subclass = os.listdir(path_subclass)

col_subclass = ['subClass', 'class', '_id', 'frequency', 'pidspread', 'pldspread']

subclass = pd.DataFrame(columns=col_subclass) # initialize empty dataframe
for file in files_subclass:
    data = pd.read_csv(path_subclass+file, sep=";")

    subclass = subclass.append(data[col_subclass], ignore_index=True)

print('{} matches for subclass'.format(len(subclass)))

subclass.to_csv('/path/to/9_FINAL/data/matches/subclass_matches.csv', sep=";")