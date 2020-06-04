import pandas as pd
import os

web_isa_folder = '/path/to/1_WebisALOD/tuplesdb_files/' # change
web_isa_files = os.listdir(web_isa_folder)

# check with subclass
subclass = pd.read_csv('/path/to/9_FINAL/data/matches_pids/subclass_with_pid.csv', sep=";")
subclass = subclass[['subClass', 'class', 'id']] # subset to relevant data

# get positive examples
positive_examples = subclass['id']

# check with transitive subclasses
transitive_subclasses = pd.read_csv('/path/to/9_FINAL/data/matches_pids/transitive-subclass_with_pid.csv',sep=";")
transitive_subclasses = transitive_subclasses[['instance', 'class', '_id']]

# get positive examples
positive_examples = positive_examples.append(transitive_subclasses['_id'])

# check with types
types = pd.read_csv('/path/to/9_FINAL/data/matches_pids/types_with_pid.csv', sep=";")
types = types[['instance', 'class', 'id']] # subset to relevant data

# get positive examples
positive_examples = positive_examples.append(types['id'])

# check with transitive types
ttypes = pd.read_csv('/path/to/9_FINAL/data/matches_pids/transitive-types_with_pid.csv', sep=";")
ttypes = ttypes[['instance', 'class', 'id']] # subset to relevant data

# get positive examples
positive_examples = positive_examples.append(ttypes['id'])
positive_examples = positive_examples.values # change to array

folderOut = '/media/linda/INTENSO/9_FINAL/data/negative_examples/'
# loop through all web is a files

checked = os.listdir(folderOut)
checked = [file.replace('negative_', '') for file in checked]

# print(len(positive_examples)) # debugging
# print(positive_examples)


### get positive subjects and objects

## subclasses
subclass_all = '/path/to/9_FINAL/data/sanitized/subclass_ontology0.csv'

data = pd.read_csv(subclass_all, sep=";") # read in data

# append subjects and objects
negative_subjects = list(data['subClass'])
negative_objects = list(data['class'])

#print(negative_objects)
#print(negative_subjects)

# print(data.columns)

# print('Length subclasses {}'.format(len(data)))

## transitive subclasses
transitive_subclasses = '/path/to9_FINAL/data/sanitized/transitive-subclasses_wo_duplicates.csv'

data = pd.read_csv(transitive_subclasses, sep=";") # read in data

# append subjects and objects
negative_subjects.extend(list(data['subclass']))
negative_objects.extend(list(data['class']))

#print(data.columns)

#print('Length subclasses {}'.format(len(data)))

#print(negative_objects)
#print(negative_subjects)

## types
types_all_folder = '/path/to/9_FINAL/data/sanitized/types/'
types_all = os.listdir(types_all_folder)

#len_types = 0 # for debugging
# get all subjects without positive examples


for file in types_all:
    data = pd.read_csv(types_all_folder+file, sep=";") # read in data
    
    # append subjects and objects
    negative_subjects.extend(list(data['instance']))
    negative_objects.extend(list(data['class']))
    
    # print(data.columns)
    

 #   len_types += len(data) # for debugging

# print('Length types {}'.format(len_types))

## transitive-types
ttypes_all_folder = '/path/to/9_FINAL/data/sanitized/transitive-types/'
ttypes_all = os.listdir(ttypes_all_folder)

#len_ttypes = 0 # for debugging
# get all subjects without positive examples

for file in ttypes_all:
    data = pd.read_csv(ttypes_all_folder+file, sep=";") # read in data

    # append subjects and objects
    negative_subjects.extend(list(data['instance']))
    negative_objects.extend(list(data['class']))

    #print(data.columns)

    #len_ttypes += len(data) # for debugging

#print('Length ttypes {}'.format(len_ttypes))

# remove duplicates
negative_subjects = list(set(negative_subjects))
negative_objects = list(set(negative_objects))

print('Found {} subjects'.format(len(negative_subjects)))
print('Found {} objects'.format(len(negative_objects)))

# subject of webisalod should be subject or object
negative_subjects.extend(negative_objects)
negative_subjects = list(set(negative_subjects))

print('Check {} for subjects'.format(len(negative_subjects)))
# folder out, account for memory error
folder_out = '/path/to/9_FINAL/data/negative_examples/'

for web_isa in web_isa_files:
    if web_isa not in checked:

        web = pd.read_csv(web_isa_folder+web_isa, sep=",")
        print('Reading file {} into memory with length {}'.format(web_isa, len(web)))

        # subset data with all observations which are not in the positive examples
        web = web[~web['_id'].isin(positive_examples)]
        print('After deletion of positive examples {}'.format(len(web)))

        # check if subject of web is alod is in caligraph as subject or object
        web = web[web['instance'].isin(negative_subjects)]
        print('After deletion of overlap for subjects {}'.format(len(web)))

        # check if object of web is alod is in caligraph as object
        web = web[web['class'].isin(negative_objects)]
        print('After deletion of overlap for objects {}'.format(len(web)))

        # export data
        web.to_csv(folder_out+'negative_'+web_isa, sep=";")

        print('-------------------------------------------')
