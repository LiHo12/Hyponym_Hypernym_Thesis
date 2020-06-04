import pandas as pd
import os
# split sentence and keep .
from nltk import PorterStemmer
import re
import bisect
from pathlib import Path

# get position of a token

def split_sentence(row):
    row = re.sub('\(|\)|#|\[|\]|%|/|=|-', '', str(row))
    return [u for x in str(row).split('.') for u in (x, '.')]
 #   return [x+'.' for x in str(row).split('.')]

def split_with_space(row, stemmer):
    sentence = []

    for element in row:
        if element != '.':
            element = element.split(' ')
        sentence.extend(element)
    # remove all whitespaces in sentence
    sentence = list(filter(lambda a: a != '', sentence))
    sentence = [stemmer.stem(str(x)) for x in sentence]

    return sentence

# get position of an element
def get_position(sentence, element, stemmer=None):
    # stem element
    if element != '.':
        element = stemmer.stem(str(element))

    return [index for index, value in enumerate(sentence) if value == element]

def check_plural(row):
    if len(row) == 0:
        return 1
    else:
        return 0

# check for singular or plural
def check_plural_index(row, sentence, element):
    if len(row) == 0:
        row = [index for index, value in enumerate(sentence) if str(element) in value]

    return row

def get_smallest_difference(subject, object):
    values = dict()
    for i in range(0, len(subject)):
        for j in range(0, len(object)):
            # get absolute value of difference
            difference = object[j] - subject[i]
            values[(i,j)] = abs(difference)
    # print(values)
    # get keys for min value
    minimum = min(values, key=values.get)

    return minimum[0], minimum[1]

def get_length_of_sentence(row, sentence, subject, object): #, instance, classname):
    # print(f'instance {instance} | class {classname}')
    # length = 8
    try:
        if len(row) == 1:
            sentence = sentence[:row[0]]
            length = len(sentence)
        elif len(row) == 0:
            length = len(sentence)
        elif len(row) == 2:
            sentence = sentence[row[0]:row[1]]
            length =  len(sentence)
        else:
            min_s, min_o = get_smallest_difference(subject, object)
            min_s = subject[min_s]
            min_o = object[min_o]

            if min_s < min_o:

                index_s = bisect.bisect(row, min_s)
                first_half = row[index_s-1]

                index_o = bisect.bisect(row, min_o)
                second_half = row[index_o]

                sentence = sentence[first_half:second_half]

            else:

                index_s = bisect.bisect(row, min_s)
                second_half = row[index_s]

                index_o = bisect.bisect(row, min_o)
                first_half = row[index_o-1]

                sentence = sentence[first_half:second_half]
            length = len(sentence)-1

        return length

    finally:
        return 8

def pad_zeros(row):
    if len(row) == 0:
        return [8]
    else:
        return row

# get folder in
folder = '/path/to/9_FINAL/data/machine_learning/two_class/distance/merged_with_contexts_'

folder_out = '/path/to/9_FINAL/data/machine_learning/two_class/sentence_length/sentence_length_'

# stem sentences
porter = PorterStemmer()

# loop through all folders
for i in range(1,78): # TODO: change number of folders
#for i in range(1,2): # debug
    exact_folder_name = folder + str(i) + '/'

    # list all files
    files = os.listdir(exact_folder_name)
    folder_out_name = folder_out + str(i) + '/'

    # create folder if not exists
    Path(folder_out_name).mkdir(parents=True, exist_ok=True)

    # get checked files
    checked_files = os.listdir(folder_out_name)
    # loop through all files in that specific folder
    for file in files:
        if file not in checked_files:
            print(file)
            # read data into memory
            data = pd.read_csv(exact_folder_name + file, sep=";", index_col=0)

            # delete pids column
            del data['pids']

            # get unique observations in terms of provid
            data = data.drop_duplicates(subset=['provid', 'id'], keep='first')

            # tokenize text
            data['sentence'] = data.sentence.apply(lambda x: split_sentence(x))
            data['sentence'] = data.sentence.apply(lambda x: split_with_space(x, porter))

            # get positions of subject and objects
            data['position_subject'] = data.apply(lambda x: get_position(x['sentence'], x['instance'], porter), axis=1)
            data['position_object'] = data.apply(lambda x: get_position(x['sentence'], x['class'], porter), axis=1)

            # get position of sentences
            data['position_sentences'] = data.apply(lambda x: get_position(x['sentence'], '.'), axis=1)

            # check if plural
            data['plural_subject'] = data.position_subject.apply(lambda x: check_plural(x))
            data['plural_object'] = data.position_object.apply(lambda x: check_plural(x))

            # pad empty positions
            data['position_subject'] = data.apply(lambda x: check_plural_index(x['position_subject'],
                                                                           x['sentence'],
                                                                           x['instance']), axis=1)
            data['position_object'] = data.apply(lambda x: check_plural_index(x['position_object'],
                                                                          x['sentence'],
                                                                          x['class']), axis=1)

            # pad if still is empty
            data['position_subject'] = data.apply(lambda x: pad_zeros(x['position_subject']), axis=1)
            data['position_object'] = data.apply(lambda x: pad_zeros(x['position_object']), axis=1)

            # get length of the sentence
            data['length_of_the_sentence'] = data.apply(lambda x: get_length_of_sentence(x['position_sentences'],
                                                                                     x['sentence'],
                                                                                     x['position_subject'],
                                                                                     x['position_object']),
                                                    axis=1)
            # debug
            # print(data)

            # export data
            data.to_csv(folder_out_name + file, sep=";", index=False)
            print('-------------------------------------------------')