import pandas as pd

# read sanitized subclasses

subclasses = pd.read_csv('/path/to/9_FINAL/data/sanitized/subclass_ontology0.csv', sep=";")

del subclasses['Unnamed: 0'] # unnecessary index

print(len(subclasses))

# get first transitive class
transitive_classes = pd.merge(subclasses, subclasses, how='left', right_on=['subClass'], left_on=['class']).dropna(subset=['class_y'])

del transitive_classes['class_x']

transitive_classes.columns = ['class_0', 'class_1', 'class_2']

transitive_classes = transitive_classes.drop_duplicates(subset=['class_0', 'class_1', 'class_2']) # drop duplicates
transitive_classes = transitive_classes.reset_index(drop=True)

print('Round 1')
print(len(transitive_classes))

# get second round of transitive classes
transitive_classes = pd.merge(transitive_classes, subclasses, how='left', left_on=['class_2'], right_on=['subClass'])

del transitive_classes['subClass'] # repeated column
transitive_classes.columns = ['class_0', 'class_1', 'class_2', 'class_3']
transitive_classes = transitive_classes.drop_duplicates(subset=['class_0', 'class_1', 'class_2', 'class_3']) # drop duplicates
print('Round 2')
print(len(transitive_classes))

# get third round of transitive classes
transitive_classes = pd.merge(transitive_classes, subclasses, how='left', left_on=['class_3'], right_on=['subClass'])

del transitive_classes['subClass'] # repeated column
transitive_classes.columns = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4']
transitive_classes = transitive_classes.drop_duplicates(subset=['class_0', 'class_1', 'class_2', 'class_3', 'class_4'])
print('Round 3')
print(len(transitive_classes))

# get fourth round of transitive classes
transitive_classes = pd.merge(transitive_classes, subclasses, how='left', left_on=['class_4'], right_on=['subClass'])

del transitive_classes['subClass'] # repeated column
transitive_classes.columns = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5']
transitive_classes = transitive_classes.drop_duplicates(subset=['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5'])
print('Round 4')
print(len(transitive_classes))

# get fifth round of transitive classes
transitive_classes = pd.merge(transitive_classes, subclasses, how='left', left_on=['class_5'], right_on=['subClass'])

del transitive_classes['subClass'] # repeated column
transitive_classes.columns = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6']
transitive_classes = transitive_classes.drop_duplicates(subset=['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6'])
print('Round 5')
print(len(transitive_classes))

# get sixth round of transitive classes
transitive_classes = pd.merge(transitive_classes, subclasses, how='left', left_on=['class_6'], right_on=['subClass'])

del transitive_classes['subClass'] # repeated column
transitive_classes.columns = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7']
transitive_classes = transitive_classes.drop_duplicates(subset=['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7'])
print('Round 6')
print(len(transitive_classes))

# get seventh round of transitive classes
transitive_classes = pd.merge(transitive_classes, subclasses, how='left', left_on=['class_7'], right_on=['subClass'])

del transitive_classes['subClass'] # repeated column
transitive_classes.columns = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7', 'class_8']
transitive_classes = transitive_classes.drop_duplicates(subset=['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7', 'class_8'])
print('Round 7')
print(len(transitive_classes))

# get eigth round of transitive classes
transitive_classes = pd.merge(transitive_classes, subclasses, how='left', left_on=['class_8'], right_on=['subClass'])

del transitive_classes['subClass'] # repeated column
transitive_classes.columns = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7', 'class_8', 'class_9']
transitive_classes = transitive_classes.drop_duplicates(subset=['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7', 'class_8', 'class_9'])
print('Round 8')
print(len(transitive_classes))

# get ninth round of transitive classes
transitive_classes = pd.merge(transitive_classes, subclasses, how='left', left_on=['class_9'], right_on=['subClass'])

del transitive_classes['subClass'] # repeated column
transitive_classes.columns = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7', 'class_8', 'class_9', 'class_10']
transitive_classes = transitive_classes.drop_duplicates(subset=['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7', 'class_8', 'class_9', 'class_10'])
print('Round 9')
print(len(transitive_classes))

# get tenth round of transitive classes
transitive_classes = pd.merge(transitive_classes, subclasses, how='left', left_on=['class_10'], right_on=['subClass'])

del transitive_classes['subClass'] # repeated column
transitive_classes.columns = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7', 'class_8', 'class_9', 'class_10', 'class_11']
transitive_classes = transitive_classes.drop_duplicates(subset=['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7', 'class_8', 'class_9', 'class_10', 'class_11'])
print('Round 10')
print(len(transitive_classes))

transitive_classes.to_csv('/path/to/9_FINAL/data/sanitized/subclass-transitive_ontology.csv', sep=";") # for debugging

