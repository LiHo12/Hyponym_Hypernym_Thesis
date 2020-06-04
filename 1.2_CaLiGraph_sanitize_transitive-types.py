import pandas as pd
import os

folderName = '/path/to/9_FINAL/data/raw/transitive-types/'
folderOut = '/path/to/9_FINAL/data/sanitized/transitive-types/'

fileList = os.listdir(folderName) # show all files in folder

print(fileList)

for file in fileList:
    if 'csv' in file:
        print("Start sanitizing file {}".format(file))
        print('-----')
        print()

        sanitizeFile = pd.read_csv(folderName + file, sep = ';')

        # delete naming column
        del sanitizeFile['Unnamed: 0']

        # lower casing
        sanitizeFile['instance'] = sanitizeFile['instance'].str.lower()
        sanitizeFile['class'] = sanitizeFile['class'].str.lower()

        # remove unnecessary elements
        sanitizeFile['instance'] = sanitizeFile['instance'].str.replace('_', ' ')
        sanitizeFile['class'] = sanitizeFile['class'].str.replace('_', ' ')

        sanitizeFile['instance'] = sanitizeFile['instance'].str.replace('owl#', '')
        sanitizeFile['class'] = sanitizeFile['class'].str.replace('owl#', '')

        sanitizeFile['instance'] = sanitizeFile['instance'].str.replace('"', '')
        sanitizeFile['class'] = sanitizeFile['class'].str.replace('"', '')

        sanitizeFile['instance'] = sanitizeFile['instance'].str.replace(':es:', '')
        sanitizeFile['class'] = sanitizeFile['class'].str.replace(':es:', '')

        sanitizeFile['instance'] = sanitizeFile['instance'].str.replace('%27', '\'')
        sanitizeFile['class'] = sanitizeFile['class'].str.replace('%27', '\'')

        sanitizeFile['instance'] = sanitizeFile['instance'].str.replace('namedindividual', 'named individual')
        sanitizeFile['class'] = sanitizeFile['class'].str.replace('namedindividual', 'named individual')

        sanitizeFile.to_csv(folderOut + file, sep = ';')

        print('Finish sanitizing {}'.format(file))
        print()
        print()