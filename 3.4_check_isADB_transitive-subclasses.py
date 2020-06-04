import pandas as pd
import os

transitive_subclasses = pd.read_csv('/path/to/9_FINAL/data/sanitized/transitive-subclasses_wo_duplicates.csv', sep=";") # read in transitive subclasses
del transitive_subclasses['Unnamed: 0'] # delete unnecessary columns
del transitive_subclasses['Unnamed: 0.1']

# get web is a files
webisa_folder = '/path/to/1_WebisALOD/tuplesdb_files/'
webisa_files = os.listdir(webisa_folder)

# save files in folder
folderOut = '/path/to/9_FINAL/data/matches/transitive-subclasses/'

# loop through webisa file
for webisa in webisa_files:
    webisa_data = pd.read_csv(webisa_folder+webisa, sep=",", encoding='utf-8', error_bad_lines=False)

    # find matches with webisa
    matches = pd.merge(webisa_data, transitive_subclasses, how='inner', left_on=['instance', 'class'], right_on=['subclass','class'])

    # found matches
    if len(matches) > 0:
        del matches['subclass'] # remove duplicate column

        webisa = webisa.replace('.csv','')
        matches.to_csv(folderOut+webisa+'_transitive-subclasses_wo_duplicates.csv',sep=";") # save files
        print('Found {} matches with {}'.format(len(matches), webisa))

    print('--------------------------------')


