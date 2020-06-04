# util functions for preprocessing during machine learning

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd

def get_train_test(data, test_size=0.05):
    '''Get train and testsplit for data.
    As default, keep 5% for testing'''
    
    return train_test_split(data, test_size=test_size, random_state=42)

def downsample_to_subclasses(subclass_train, types_train, negatives_train):
    """Downsample to the smallest class"""
    
    # subclasses remains untouched, thus only keep length
    subclasses_proportion = len(subclass_train)
    # print(subclasses_proportion) # debugging
    
    # get proportion of data
    types_train_sampled = types_train.copy().iloc[:subclasses_proportion]   
    negatives_train_sampled = negatives_train.copy().iloc[:subclasses_proportion]
    
    # get difference between three columns
    difference_column = str(set(negatives_train_sampled.columns[7:]).difference(subclass_train.columns[7:])).replace('{', '')
    difference_column = difference_column.replace('}', '')
    difference_column = difference_column.replace('\'', '')
    # print(difference_column) # for debugging
    
    # get column position
    column_position = types_train_sampled.columns.get_loc(difference_column)
    #print(column_position)
        
    # repeat zero for pattern x times
    zero_pattern = [0] * subclasses_proportion
    subclasses_new = subclass_train.copy()
    subclasses_new.insert(loc=column_position, column=difference_column, value=zero_pattern)
    
    # include label to data 
    subclasses_new['label'] = [0] * subclasses_proportion
    types_train_sampled['label'] = [1] * subclasses_proportion
    negatives_train_sampled['label'] = [2] * subclasses_proportion
    
    # change order of id column
    id_negative = negatives_train_sampled['_id']
    del negatives_train_sampled['_id']
    negatives_train_sampled.insert(loc=types_train_sampled.columns.get_loc('id'), column='id', value=id_negative)

    # rename all columns
    subclasses_new.columns = types_train_sampled.columns
    negatives_train_sampled.columns = types_train_sampled.columns
    
    # shuffle data 
    all_training_data = pd.concat([negatives_train_sampled, types_train_sampled, subclasses_new])
    all_training_data = all_training_data.reset_index(drop=True)
    all_training_data = shuffle(all_training_data)

    # for debugging
    # print('Shape subclasses: {} || shape types: {} || shape negatives: {}'.format(subclass_train.shape, types_train_sampled.shape, types_train_sampled.shape))
    #p rint(all_training_data.shape)
    # print(all_training_data.head())
    
    return all_training_data

def get_testing_set(subclass_test, types_test, negatives_test):
    """ generate column with missing pid for subclasses,
        append label,
        concat the three properties,
        shuffle data
        """
    
    # get difference between three columns
    difference_column = str(set(negatives_test.columns[7:]).difference(subclass_test.columns[7:])).replace('{', '')
    difference_column = difference_column.replace('}', '')
    difference_column = difference_column.replace('\'', '')
    # print(difference_column)
    
    # get column position
    column_position = types_test.columns.get_loc(difference_column)
    
    # repeat zero for pattern x times
    zero_pattern = [0] * len(subclass_test)
    subclasses_new = subclass_test.copy()
    subclasses_new.insert(loc=column_position, column=difference_column, value=zero_pattern)
    
    # get copies to not change original data
    types_new = types_test.copy()
    negatives_new = negatives_test.copy()
    
    # include label to data 
    subclasses_new['label'] = [0] * len(subclass_test)
    types_new['label'] = [1] * len(types_test)
    negatives_new['label'] = [2] * len(negatives_test)
    
    # change order of id column
    id_negative = negatives_new['_id']
    del negatives_new['_id']
    negatives_new.insert(loc=types_new.columns.get_loc('id'), column='id', value=id_negative)
    
    # name columns all equally
    subclasses_new.columns = types_new.columns
    negatives_new.columns = types_new.columns
        
    # stack data all together and reset index
    all_testing_data = pd.concat([subclasses_new, types_new, negatives_new]).reset_index(drop=True)
    
    # shuffle data
    all_testing_data = shuffle(all_testing_data)
    
    return all_testing_data

def upsample_to_types(subclass_train, types_train, negatives_train):
    '''Upsample to type class'''
    # get length of types
    types_length = len(types_train)
    # print(types_length)
    
    # downsample negatives, upsample types
    negatives_train_sampled = negatives_train.copy().iloc[:types_length]
    types_train_sampled = types_train.copy()
    # print(len(negatives_train_sampled))
    # copy x times subclasses until it has the same size as types
    subclasses_train_sampled = subclass_train.copy() # get copy
    copy_subclasses = int(types_length/len(subclasses_train_sampled))-1
    
    # copy data of minority class
    counter = 0
    while counter < copy_subclasses:
        subclasses_train_sampled = pd.concat([subclasses_train_sampled, subclass_train])
        counter += 1
    
    rest_subclasses = subclass_train.iloc[:types_length-len(subclasses_train_sampled)]
    subclasses_train_sampled = pd.concat([subclasses_train_sampled, rest_subclasses])
    # print(len(subclasses_train_sampled))
    
     # get difference between three columns
    difference_column = str(set(negatives_train_sampled.columns[7:]).difference(subclass_train.columns[7:])).replace('{', '')
    difference_column = difference_column.replace('}', '')
    difference_column = difference_column.replace('\'', '')
    # print(difference_column) # for debugging
    
    # get column position
    column_position = types_train_sampled.columns.get_loc(difference_column)
    
    # repeat zero for pattern x times
    zero_pattern = [0] * types_length
    subclasses_train_sampled.insert(loc=column_position, column=difference_column, value=zero_pattern)
    
    # include label to data 
    subclasses_train_sampled['label'] = [0] * types_length
    types_train_sampled['label'] = [1] * types_length
    negatives_train_sampled['label'] = [2] * types_length
    
    # change order of id column
    id_negative = negatives_train_sampled['_id']
    del negatives_train_sampled['_id']
    negatives_train_sampled.insert(loc=types_train_sampled.columns.get_loc('id'), column='id', value=id_negative)

    # rename all columns
    subclasses_train_sampled.columns = types_train_sampled.columns
    negatives_train_sampled.columns = types_train_sampled.columns
    
    # shuffle data 
    all_training_data = pd.concat([negatives_train_sampled, types_train_sampled, subclasses_train_sampled])
    all_training_data = all_training_data.reset_index(drop=True)
    all_training_data = shuffle(all_training_data)
    
    return all_training_data