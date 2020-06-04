import pandas as pd
import numpy as np
import os

folderName = '/path/to/9_FINAL/data/raw/subclass/'

fileList = os.listdir(folderName) # show all files in folder

print(fileList)

for file in fileList:
    if ('csv' in file):
        print("Start sanitizing file {}".format(file))
        print('-----')
        print()

        sanitizeFile = pd.read_csv(folderName + file, sep = ';')

        # delete naming column
        del sanitizeFile['Unnamed: 0']

        # lower casing
        sanitizeFile['subClass'] = sanitizeFile['subClass'].str.lower()
        sanitizeFile['class'] = sanitizeFile['class'].str.lower()

        # remove unnecessary elements
        sanitizeFile['subClass'] = sanitizeFile['subClass'].str.replace('_', ' ')
        sanitizeFile['class'] = sanitizeFile['class'].str.replace('_', ' ')

        sanitizeFile['subClass'] = sanitizeFile['subClass'].str.replace('owl#', '')
        sanitizeFile['class'] = sanitizeFile['class'].str.replace('owl#', '')

        sanitizeFile['subClass'] = sanitizeFile['subClass'].str.replace('"', '')
        sanitizeFile['class'] = sanitizeFile['class'].str.replace('"', '')

        sanitizeFile['subClass'] = sanitizeFile['subClass'].str.replace(':es:', '')
        sanitizeFile['class'] = sanitizeFile['class'].str.replace(':es:', '')

        sanitizeFile['subClass'] = sanitizeFile['subClass'].str.replace('%27', '\'')
        sanitizeFile['class'] = sanitizeFile['class'].str.replace('%27', '\'')

        sanitizeFile['subClass'] = sanitizeFile['subClass'].str.replace('namedindividual', 'named individual')
        sanitizeFile['class'] = sanitizeFile['class'].str.replace('namedindividual', 'named individual')

        sanitizeFile.to_csv(folderName + 'sanitized/' + file, sep = ';')

        print('Finish sanitizing {}'.format(file))
        print()
        print()