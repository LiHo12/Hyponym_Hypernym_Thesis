import os
import pandas as pd

# there are two knowledge graph relationships: subClasses & (transitive-)types
# this file checks all instances with the the instances in webisALOD

folderNameWeb = '/path/to/1_WebisALOD/tuplesdb_files/' # on hard drive
folderNameInstances = '/path/to/9_FINAL/data/sanitized/subclass_ontology0.csv'

webs = os.listdir(folderNameWeb)

folderNameOut = '/path/to/9_FINAL/data/matches/subclass/' # save matches

matches = 0

instances = pd.read_csv(folderNameInstances, sep=";")

for web in webs:
    print('Start to compare with {}'.format(web))

    compare = pd.read_csv(folderNameWeb + web, sep=',', encoding='utf-8', error_bad_lines=False)

    del compare['modifications']

    # matches by merge
    match = pd.merge(instances, compare, how='inner', left_on=['subClass', 'class'], right_on=['instance', 'class'])

    len_match = len(match)

    if len_match > 0:
        # found matches
        print('Successfully found {} matches with {}'.format(len_match, web))
        matches += len_match
        outName = web.replace('.csv', '_')
        match.to_csv(folderNameOut + outName + 'subclass_ontology0.csv', sep=";")

    print('------------------------------------')

print('Found {} matches in total.'.format(matches))

