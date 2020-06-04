import pandas as pd
import re
from nltk.stem import PorterStemmer
import os
from pathlib import Path

def get_index(string_array, value_detail):
    # get index of pattern, account for noise with signs
    # print(value_detail)
    try:
        pattern = re.compile(value_detail+'\\b')
      #  print(pattern)
        indices  = [index for index, value in enumerate(string_array) if pattern.match(value)]
    except:
        indices = [index for index, value in enumerate(string_array) if value == value_detail]

    return indices

def get_distance_of_instance_and_class(row, stemmer):
    # calculate distance between instance and class

    # stem sentence and split sentence to get index
    sentence_check = str(row['sentence']).split(' ')
    sentence_check = [stemmer.stem(str(x)) for x in sentence_check]
    #print(sentence_check)
    #print('Instance: ' + row['instance'])

    # get index for instance
    instance_stemmed = stemmer.stem(str(row['instance']))

    #print('Instance: ' + row['instance'])
    #print('stemmed instance: ' + instance_stemmed)

    index_instance = get_index(sentence_check, instance_stemmed)
    #print('Indices: ' + str(index_instance))

    # get index for class
    class_stemmed = stemmer.stem(str(row['class']))

    #print(class_stemmed)
    index_class = get_index(sentence_check, class_stemmed)
    #print('stemmed class: ' + class_stemmed)
    #print('Class: ' + row['class'])
    #print('Indices: '+ str(index_class))

    # calculate difference with each class
    differences = []
    for element in index_class:
        # calculate position difference between instance and class
        difference = [element - x for x in index_instance]
        differences.extend(difference)

    #print(row['position'])
    #print('Differences: ' + str(differences))

    # check if position positive or negative
    if len(differences) > 0:
        #min_differences = 0
        if row['position'] == 1:
            # check if positive and negatives in array
            differences = [x for x in differences if x >= 0]
            if len(differences) > 0:
                min_differences = min(differences)
            else:
                min_differences = 2

        else:
            differences = [x for x in differences if x < 0]
            if len(differences) > 0:
                min_differences = max(differences)
            else:
                min_differences = -2
    else:
        min_differences = 2 # default case

    return min_differences

# stem sentences
porter = PorterStemmer()

# read in pattern details for sanity check
pattern_details = pd.read_csv('/path/to/9_FINAL/data/context/pattern_details_position.csv',sep=";",
                              index_col=0)
# get folder
folder = '/path/to/9_FINAL/data/goldstandard/contexts/'
folder_out = '/path/to/9_FINAL/data/goldstandard/contexts_calculated/'

# get files in folder
files = os.listdir(folder)

# loop through all files
for file in files:
    # debug
    print(f'Start with file {file}')
    data = pd.read_csv(folder + file, sep=";")

    # match with patterns
    data = pd.merge(data, pattern_details, how='left',
                    right_on='pids', left_on='pids')

    # replace all signs in sentence column
    data['sentence'] = data.sentence.str.replace(',', '')
    data['sentence'] = data.sentence.str.replace(';', '')
    data['sentence'] = data.sentence.str.replace('.', '')
    data['sentence'] = data.sentence.str.replace(':', '')

    data['distance'] = data.apply(lambda x: get_distance_of_instance_and_class(x, porter), axis=1)

    # save memory space
    del data['sentence']

    # export file
    data.to_csv(folder_out+file, sep=";", index=False)
