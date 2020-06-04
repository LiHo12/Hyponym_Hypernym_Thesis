import pandas as pd
import os


# explode entries test
folder = '/path/to/9_FINAL/data/machine_learning/two_class/distance/rest_explode/'
files = os.listdir(folder)

checked_files = os.listdir('/path/to/9_FINAL/data/machine_learning/two_class/distance/exploded_pid_provids/')

# loop through all files and explode entries
for file in files:
    print('Start parsing file {}'.format(file))

    data = pd.read_csv(folder+file, sep=";")

    if data.columns[1] != '_id':
            #print(data.head())
        data.columns = ['Unnamed: 0', '_id', 'instance', 'class', 'provids']

        # subset data
        # data = data[['_id', 'provids']]
    del data['Unnamed: 0']



        # replace unnecessary characters
    data['provids'] = [x.replace('\'', '') for x in data['provids']]
    data['provids'] = [x.replace('[', '') for x in data['provids']]
    data['provids'] = [x.replace(']', '') for x in data['provids']]
    data['provids'] = [x.replace(' ', '') for x in data['provids']]


        # explode data
    exploded_data = pd.DataFrame(data.provids.str.split(',').tolist(), index=[data['_id'], data['instance'], data['class']]).stack()
    exploded_data = exploded_data.reset_index([0, '_id', 'instance', 'class'])  # var1 variable is currently labeled 0
    exploded_data.columns = ['id', 'instance', 'class', 'provid']
    exploded_data = exploded_data[exploded_data.provid != ''] # delete empty observations
    exploded_data = exploded_data.reset_index(drop=True)
        # print(exploded_data.head())

        # get pid column and provid
    exploded_data['pids'] = exploded_data.provid.apply(lambda x: str(x).split(';patterns:')[1])
    exploded_data['provid'] = exploded_data.provid.apply(lambda x: str(x).split('patterns:')[0])

        # export data
    exploded_data.to_csv('/path/to/9_FINAL/data/machine_learning/two_class/distance/exploded_pid_provids/'+file, sep=";")

    print('------------------------------')
