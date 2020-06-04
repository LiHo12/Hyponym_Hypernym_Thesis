import pandas as pd
import os

# the transitive data has the following structure:
## class_0, ...., class_10
## the transitive relationship always skips the second class, i.e.
## class_0 is transitive towards class_2, ..., class_10
## class_1 is transitive towards class_3, ..., class_10

# read in data
transitive_subclass_folder = '/path/to/9_FINAL/data/sanitized/subclass-transitive_ontology.csv'
transitive_subclasses = pd.read_csv(transitive_subclass_folder, sep=";")

# initialize empty arrays
subclasses = []
classes = []

# loop through each row of dataframe and extend view to transitive relationships
for index, row in transitive_subclasses.iterrows():
    # get all transitive relationships in one row

    counter_0 = 0
    # keep first one column as constant
    while counter_0 <= 9:

        subclass_name = 'class_' + str(counter_0)
        #print(subclass_name)  # for debugging

        # loop through transitive columns
        counter_1 = counter_0 + 2

        subclass = row[subclass_name]

        if isinstance(subclass, float):
            #print('--------')
            break

        if isinstance(subclass, str):
            while counter_1 <= 11:
                class_name = 'class_' + str(counter_1)
                # print(class_name)
                class_ = row[class_name]

                # account for empty cases and bugs
                if isinstance(class_, str):
                    print('subclass {} || class: {}'.format(subclass, class_))
                    subclasses.append(subclass)
                    classes.append(class_)
                if isinstance(class_, float):
                    # for nan
                    #print('--------')
                    break # stop showing all classes
                counter_1 += 1

        counter_0 += 1
        #print('--------')

transitive_matches = pd.DataFrame({'subclass': subclasses,
                                   'class': classes})

transitive_matches.to_csv('/path/to/9_FINAL/data/sanitized/transitive-subclasses.csv', sep=";")