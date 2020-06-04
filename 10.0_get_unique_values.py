import pandas as pd

# check whether overlapping folds
training = '/path/to/9_FINAL/data/machine_learning/two_class/one-hot-ecoding/train/cross_validation/training/training_'
validation = '/path/to/9_FINAL/data/machine_learning/two_class/one-hot-ecoding/train/cross_validation/validation/val_fold_'

all_data = []

# get unique values
for x in range(0,5):
    training_name = training+str(x)+'.csv'
    training_file = pd.read_csv(training_name, sep=";") # read data
    training_file = training_file[['instance', 'class', 'id', 'label']] # subset to relevant columns
    all_data.append(training_file) # append data
    print('Training: {}'.format(len(training_file))) # sanity check

    validation_name = validation + str(x) + '.csv'
    validation_file = pd.read_csv(validation_name,sep=";")#
    validation_file = validation_file[['instance', 'class', 'id', 'label']]
    all_data.append(validation_file)
    print('Validation: {}'.format(len(validation_file)))

    unique_ids = training_file.id.tolist()
    unique_ids.extend(validation_file.id.tolist())
    unique_ids = list(set(unique_ids))
    print('Unique values: {}'.format(len(unique_ids)))
    print('All values {}'.format(len(training_file)+len(validation_file)))
    print('----------------------------------------')

all_data = pd.concat(all_data)
print(all_data.head())
all_data.columns = ['instance', 'class', 'id', 'label']

# remove duplicates
all_data = all_data.drop_duplicates(subset='id', keep='first')

print(len(all_data))

all_data.to_csv('/path/to/9_FINAL/data/machine_learning/two_class/distance/raw/all_distance_tuples.csv',sep=";")