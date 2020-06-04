import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, plot_confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, BorderlineSMOTE

def train_ml(model, X_train, y_train, X_validation):
    '''Train on train set and predict on validation set'''
    model.fit(X_train, y_train)
    
    return model.predict(X_validation)


def read_file_and_drop_columns(file_name, columns_to_drop):
    '''Read file and drop the interesting columns'''
    # read data
    data = pd.read_csv(file_name, sep=";")
    
    if 'Unnamed: 0' in data.columns.to_list():
        del data['Unnamed: 0']
        
    # drop columns
    data.drop(columns_to_drop, axis=1, inplace=True)
    
    # get label and delete from data
    label = data['label']
    del data['label']
    
    return data, label


def get_metrics(y_predicted, y_true):
    '''Get all relevant metrics:
    - F1-Macro
    - Precision
    - Recall
    - Confusion Matric'''
    
    precision = precision_score(y_true, y_predicted)
    recall = recall_score(y_true, y_predicted)
    f1 = f1_score(y_true, y_predicted)
    
    conf_matrix = confusion_matrix(y_true, y_predicted)
    
    # print confusion matrix, micro and macro
    print(conf_matrix)
    print(f'Precision {precision} and Recall {recall} and F1 {f1}')
    print('----------------------------------')
    
    return precision, recall, f1, conf_matrix


def cross_validation(validation_folder, train_folder, columns_to_drop,
                    model, scale=False, sample=False):
    '''Cross validate 5 times and get overall F1 macro and micro average'''
    # initialize empty lists for overall values
    all_precisions = []
    all_recalls = []
    all_f1s = []
    
    # negatives
    overall_negative_precision = []
    overall_negative_recall = []
    overall_negative_f1 = []
    
    for i in range(0,5):
        # get validation and training folder
        train_file = train_folder + str(i) + '.csv'
        validation_file = validation_folder + str(i) + '.csv'
        
        # get train and validation file
        X_train, y_train = read_file_and_drop_columns(train_file, columns_to_drop)
        #print(X_train.columns)
        X_validation, y_validation = read_file_and_drop_columns(validation_file, columns_to_drop)
        #print(X_validation.columns)
        # scale if True given
        if scale:
            scaler = MinMaxScaler()
            # scale train and validation
            X_train = scaler.fit_transform(X_train)
            X_validation = scaler.fit_transform(X_validation)
        
        # sample if sample = True
        if sample:
            oversample = SMOTE(sampling_strategy=0.2,
                          k_neighbors=5)
            X_train, y_train = oversample.fit_resample(X_train, y_train)
            undersample = RandomUnderSampler(sampling_strategy=0.5)
            X_train, y_train = undersample.fit_resample(X_train, y_train)
        
        # train and predict
        y_predicted = train_ml(model, X_train, y_train, X_validation)
        # print(y_predicted)
        
        # predict and get micro & macro
        precision, recall, f1, conf_matrix = get_metrics(y_predicted, y_validation)
        
        # append values for overall measures
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)
        
        # flip for negatives
        precision = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[1,0])
        recall = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[0,1])
        f1 = (2*precision*recall)/(precision+recall)
        
        overall_negative_precision.append(precision)
        overall_negative_recall.append(recall)
        overall_negative_f1.append(f1)
        
    # print mean of the values
    print('---------------------')
    print(f'Overall Precision: {np.mean(all_precisions)}')
    print(f'Overall Recall: {np.mean(all_recalls)}')
    print(f'Overall F1: {np.mean(all_f1s)}')
    
    # print mean of the values for negatives
    print('---------------------')
    print(f'Negative Precision: {np.mean(overall_negative_precision)}')
    print(f'Negative Recall: {np.mean(overall_negative_recall)}')
    print(f'Negative F1: {np.mean(overall_negative_f1)}')
    