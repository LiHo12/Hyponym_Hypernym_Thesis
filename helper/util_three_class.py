import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, plot_confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def train_ml(model, X_train, y_train, X_validation):
    '''Train on train set and predict on validation set'''
    model.fit(X_train, y_train)
    
    return model.predict(X_validation)


def read_file_and_drop_columns(file_name, columns_to_drop):
    '''Read file and drop the interesting columns'''
    # read data
    data = pd.read_csv(file_name, sep=";", index_col=0)
    
    # drop columns
    data.drop(columns_to_drop, axis=1, inplace=True)
    
    # get label and delete from data
    label = data['label']
    del data['label']
    
    return data, label


def get_metrics(y_predicted, y_true):
    '''Get all relevant metrics:
    - F1-Macro
    - F1-Micro
    - Confusion Matric'''
    
    micro = f1_score(y_true, y_predicted, average='micro')
    macro = f1_score(y_true, y_predicted, average='macro')
    
    # print confusion matrix, micro and macro
    print(confusion_matrix(y_true, y_predicted))
    print(f'Micro {micro} and Macro {macro}')
    print('----------------------------------')
    
    return micro, macro


def cross_validation(validation_folder, train_folder, columns_to_drop,
                    model, scale=False):
    '''Cross validate 5 times and get overall F1 macro and micro average'''
    # initialize empty lists for overall values
    all_micros = []
    all_macros= []
    
    for i in range(0,5):
        # get validation and training folder
        train_file = train_folder + str(i) + '.csv'
        validation_file = validation_folder + str(i) + '.csv'
        
        # get train and validation file
        X_train, y_train = read_file_and_drop_columns(train_file, columns_to_drop)
        X_validation, y_validation = read_file_and_drop_columns(validation_file, columns_to_drop)
        
        # scale if True given
        if scale:
            scaler = MinMaxScaler()
            # scale train and validation
            X_train = scaler.fit_transform(X_train)
            X_validation = scaler.fit_transform(X_validation)
        
        # train and predict
        y_predicted = train_ml(model, X_train, y_train, X_validation)
        # print(y_predicted)
        
        # predict and get micro & macro
        micro, macro = get_metrics(y_predicted, y_validation)
        
        # append values for overall measures
        all_micros.append(micro)
        all_macros.append(macro)
        
    # print mean of the values
    print('---------------------')
    print(f'Overall Micro: {np.mean(all_micros)}')
    print(f'Overall Macro: {np.mean(all_macros)}')