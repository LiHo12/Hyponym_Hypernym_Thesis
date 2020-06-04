import os
import pandas as pd

# there are two knowledge graph relationships: subClasses & (transitive-)types
# this file checks all instances with the the instances in webisALOD

folderNameWeb = '/path/to/1_WebisALOD/tuplesdb_files/' # on hard drive
folderNameInstances = '/path/to/9_FINAL/data/sanitized/transitive-types/'

webs = os.listdir(folderNameWeb)
instances = os.listdir(folderNameInstances)

folderNameOut = '/path/to/9_FINAL/data/matches/transitive-types/' # save matches

matches = 0

# loop through all files in instances:
for instance in instances:
        # read data
    data = pd.read_table(folderNameInstances + instance, sep = ';', encoding = 'utf-8', error_bad_lines=False)

    print('Start comparing file {}'.format(instance))

    for web in webs:
        compare = pd.read_csv(folderNameWeb + web, sep = ',', encoding='utf-8', error_bad_lines=False)

        del compare['modifications']

            # matches by merge
        match = pd.merge(data, compare, how='inner', left_on=['instance', 'class'], right_on=['instance', 'class'])

        len_match = len(match)

        if len_match > 0:
            # found matches
            print('Successfully found {} matches with {}'.format(len_match, web))
            matches += len_match
            outName = web.replace('.csv', '_')
            match.to_csv(folderNameOut+outName+instance, sep=";")

        print('------------------------------------')
        print('Finished with file {}'.format(instance))
        print('------------------------------------')

print('Found {} matches in total.'.format(matches))
