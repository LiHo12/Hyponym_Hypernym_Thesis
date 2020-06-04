import pandas as pd

transitive_subclasses = pd.read_csv('/path/to/9_FINAL/data/sanitized/transitive-subclasses.csv',sep=";")
print('Length before removal of duplicates: {}'.format(len(transitive_subclasses)))

transitive_subclasses = transitive_subclasses.drop_duplicates(subset=['subclass','class']) # drop duplicates
transitive_subclasses = transitive_subclasses.reset_index(drop=True) # reset indices
print('Length after removal of duplicates: {}'.format(len(transitive_subclasses)))

indices = []
for index, row in transitive_subclasses.iterrows():
    if row['subclass'] == row['class']:
        print('{} || {}'.format(row['subclass'], row['class']))
        print('Index: {}'.format(index))
        indices.append(index)

transitive_subclasses = transitive_subclasses[~transitive_subclasses.index.isin(indices)]
transitive_subclasses = transitive_subclasses.reset_index(drop=True) # reset indices
print('Length after removal of equal subclass and class: {}'.format(len(transitive_subclasses)))

transitive_subclasses.to_csv('/path/to/9_FINAL/data/sanitized/transitive-subclasses_wo_duplicates.csv',sep=";")