import pandas as pd

### subclasses
subclasses = pd.read_csv('/path/to/9_FINAL/data/matches/subclass_matches.csv', sep=";")
subclasses['appended'] = subclasses['subClass'] + ' ' + subclasses['class']
#subclasses['appended2'] = subclasses['subClass'] + ' ' + subclasses['class'] + ' ' + subclasses['_id'].astype(str) + ' ' + subclasses['pidspread'].astype(str) + ' '+ subclasses['pldspread'].astype(str)
print(subclasses['appended'].value_counts()) # duplicates aids person + aids aids # 2
print(len(subclasses))
#print(subclasses['appended2'].value_counts()) # duplicates aids person + aids aids # 2 same id, pidspread & pldspread

# drop duplicates
subclasses = subclasses.drop_duplicates(['appended']).reset_index(drop=True)
print(len(subclasses)) # 1,692

del subclasses['Unnamed: 0']
del subclasses['appended']
subclasses.to_csv('/path/to/9_FINAL/data/matches/subclass_matches_wo_duplicates.csv', sep=";")

### transitive-types
ttypes = pd.read_csv('/path/to/9_FINAL/data/matches/transitive-types_matches.csv', sep=";")
ttypes['appended'] = ttypes['instance'] + ' ' + ttypes['class']
ttypes['appended2'] = ttypes['instance'] + ' ' + ttypes['class'] + ' ' + ttypes['_id'].astype(str) + ' ' + ttypes['pidspread'].astype(str) + ' '+ ttypes['pldspread'].astype(str)
print(len(ttypes))

values = pd.DataFrame(ttypes['appended'].value_counts())
print(values[values['appended'] > 1]) # 486 values
values = pd.DataFrame(ttypes['appended2'].value_counts())
print(values[values['appended2'] > 1]) # 493 values

ttypes = ttypes.drop_duplicates('appended2').reset_index(drop=True)

del ttypes['Unnamed: 0']
del ttypes['appended']
del ttypes['appended2']

ttypes.to_csv('/path/to/9_FINAL/data/matches/transitive-types_matches_wo_duplicates.csv', sep=";")
print(len(ttypes))

### types
types = pd.read_csv('/path/to/9_FINAL/data/matches/types_matches.csv', sep=";")
types['appended'] = types['instance'] + ' ' + types['class']
# types['appended2'] = types['instance'] + ' ' + types['class'] + ' ' + types['_id'].astype(str) + ' ' + types['pidspread'].astype(str) + ' '+ types['pldspread'].astype(str)

print(len(types))

values = pd.DataFrame(types['appended'].value_counts())
values = values[values['appended'] > 1]
print(len(values)) # 23 values

# values = pd.DataFrame(types['appended2'].value_counts())
# values = values[values['appended2'] > 1]
# print(len(values)) # 23 values

types = types.drop_duplicates('appended').reset_index(drop=True)

del types['Unnamed: 0']
del types['appended']

types.to_csv('/path/to/9_FINAL/data/matches/types_matches_wo_duplicates.csv', sep=";")
print(len(types))