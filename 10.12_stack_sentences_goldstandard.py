import pandas as pd
import os

# get all files in folder
folder = '/path/to/9_FINAL/data/goldstandard/contexts/'
files = os.listdir(folder)

# initialize empty data frame
all_data = pd.DataFrame()

# loop through files and stack together
for file in files:

    data = pd.read_csv(folder+file, sep=";")

    all_data = pd.concat([all_data, data], ignore_index=True)

# export data
all_data.to_csv('/path/to/9_FINAL/data/goldstandard/goldstandard_with_sentences.csv',
                index=False, sep=";")

# subset data to positives
positives_id = ['516170918', '105870566', '63424816', '139132257', '209693125']
positives = all_data[all_data.id.isin(positives_id)]
positives.to_csv('/path/to/9_FINAL/data/goldstandard/goldstandard_positives_with_sentences.csv',
                index=False, sep=";")

# subset data to negatives
negatives = all_data[~all_data.id.isin(positives_id)]
negatives.to_csv('/path/to/9_FINAL/data/goldstandard/goldstandard_negatives_with_sentences.csv',
                index=False, sep=";")