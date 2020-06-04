# util functions for machine learning

import pandas as pd
from sklearn import metrics
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
import numpy as np
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, f1_score, precision_score, recall_score
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, BorderlineSMOTE

def cross_validate_algorithm(algorithm, X_train, y_train,
                            scoring, folds=5):
    """Get cross validated score for both macro and micro"""
    
    scores = cross_val_score(algorithm, X_train, y_train, cv=folds, scoring=scoring)
    print(scores)
    print('{} score: {} with std {}'.format(scoring, scores.mean, scores.std()*2))
    
def get_cross_validated_confusion_matrix(model, X_train, y_train, folds=5, sample=False):
    """Get cross validated confusion matrix"""
    # k-fold
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=88)
    kf.get_n_splits(X_train.values)
    
    overall_macro = []
    overall_micro = []
    
    for train_index, test_index in kf.split(X_train.values, y_train):
        # print("TRAIN:", train_index, "TEST:", test_index) # for debugging
        X_tr, X_te = X_train.values[train_index], X_train.values[test_index]
        y_tr, y_te = y_train[train_index], y_train[test_index]
        
        # sample if sample = True
        if sample:
            oversample = SMOTE(sampling_strategy=0.1,
                          k_neighbors=5)
            X_tr, y_tr = oversample.fit_resample(X_tr, y_tr)
            undersample = RandomUnderSampler(sampling_strategy=0.5)
            X_tr, y_tr = undersample.fit_resample(X_tr, y_tr)

        # fit model
        model.fit(X_tr, y_tr)
        
        # predicted score
        y_predict = model.predict(X_te)
        micro = f1_score(y_te, y_predict, average='micro')
        macro = f1_score(y_te, y_predict, average='macro')
        
        print(confusion_matrix(y_te, y_predict))
        print('Macro: {} || Micro: {}'.format(macro, micro))
        
        overall_macro.append(macro)
        overall_micro.append(micro)
    
    print('---------------------------')
    print('Overall Macro: {} (+/- {}) || Overall Micro: {} (+/- {})'.format(np.mean(overall_macro), np.std(overall_macro, axis=0),
                                                                           np.mean(overall_micro), np.std(overall_micro, axis=0)))
    
    
def get_cross_validated_two_class_confusion_matrix(model, X_train, y_train, folds=5):
    """Get cross validated confusion matrix for two class problem"""
    # k-fold
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=88)
    kf.get_n_splits(X_train.values)
    
    overall_precision = []
    overall_recall = []
    overall_f1 = []
    
    # negative scores
    overall_negative_precision = []
    overall_negative_recall = []
    overall_negative_f1 = []
    
    for train_index, test_index in kf.split(X_train.values, y_train):
        # print("TRAIN:", train_index, "TEST:", test_index) # for debugging
        X_tr, X_te = X_train.values[train_index], X_train.values[test_index]
        y_tr, y_te = y_train[train_index], y_train[test_index]
        
        # fit model
        model.fit(X_tr, y_tr)
        
        # predicted score
        y_predict = model.predict(X_te)
        f1 = f1_score(y_te, y_predict)
        precision = precision_score(y_te, y_predict)
        recall = recall_score(y_te, y_predict)
        conf_matrix = confusion_matrix(y_te, y_predict)
        print(conf_matrix)
        print('Positive')
        print('Precision: {} || Recall: {} || F1: {}'.format(precision, recall, f1))
        
        overall_precision.append(precision)
        overall_recall.append(recall)
        overall_f1.append(f1)
        
        # for negative class (noise), flip labels
        
        precision = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[1,0])
        recall = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[0,1])
        f1 = (2*precision*recall)/(precision+recall)
        
        print('Negative')
        print('Precision: {} || Recall: {} || F1: {}'.format(precision, recall, f1))
        
        overall_negative_precision.append(precision)
        overall_negative_recall.append(recall)
        overall_negative_f1.append(f1)
    
    print('---------------------------')
    print('Positive')
    print('Overall Precision: {} (+/- {}) || Overall Recall: {} (+/- {}) || Overall F1: {} (+/- {})'.format(
                                                                            np.mean(overall_precision), np.std(overall_precision, axis=0),
                                                                           np.mean(overall_recall), np.std(overall_recall, axis=0),
                                                                            np.mean(overall_f1), np.std(overall_f1, axis=0)))
    
    print('Negative')
    print('Overall Precision: {} (+/- {}) || Overall Recall: {} (+/- {}) || Overall F1: {} (+/- {})'.format(
                                                                            np.mean(overall_negative_precision), np.std(overall_negative_precision, axis=0),
                                                                           np.mean(overall_negative_recall), np.std(overall_negative_recall, axis=0),
                                                                            np.mean(overall_negative_f1), np.std(overall_negative_f1, axis=0)))
    
def get_cross_validated_normal_two_class_confusion_matrix(model, X_train, y_train, folds=5):
    """Get cross validated confusion matrix for two class problem"""
    # k-fold
    kf = KFold(n_splits=folds, shuffle=True, random_state=88)
    kf.get_n_splits(X_train.values)
    
    overall_precision = []
    overall_recall = []
    overall_f1 = []
    
    # negative scores
    overall_negative_precision = []
    overall_negative_recall = []
    overall_negative_f1 = []
    
    for train_index, test_index in kf.split(X_train.values):
        # print("TRAIN:", train_index, "TEST:", test_index) # for debugging
        X_tr, X_te = X_train.values[train_index], X_train.values[test_index]
        y_tr, y_te = y_train[train_index], y_train[test_index]
        
        # fit model
        model.fit(X_tr, y_tr)
        
        # predicted score
        y_predict = model.predict(X_te)
        f1 = f1_score(y_te, y_predict)
        precision = precision_score(y_te, y_predict)
        recall = recall_score(y_te, y_predict)
        conf_matrix = confusion_matrix(y_te, y_predict)
        print(conf_matrix)
        print('Positive')
        print('Precision: {} || Recall: {} || F1: {}'.format(precision, recall, f1))
        
        overall_precision.append(precision)
        overall_recall.append(recall)
        overall_f1.append(f1)
        
        # for negative class (noise), flip labels
        
        precision = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[1,0])
        recall = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[0,1])
        f1 = (2*precision*recall)/(precision+recall)
        
        print('Negative')
        print('Precision: {} || Recall: {} || F1: {}'.format(precision, recall, f1))
        
        overall_negative_precision.append(precision)
        overall_negative_recall.append(recall)
        overall_negative_f1.append(f1)
    
    print('---------------------------')
    print('Positive')
    print('Overall Precision: {} (+/- {}) || Overall Recall: {} (+/- {}) || Overall F1: {} (+/- {})'.format(
                                                                            np.mean(overall_precision), np.std(overall_precision, axis=0),
                                                                           np.mean(overall_recall), np.std(overall_recall, axis=0),
                                                                            np.mean(overall_f1), np.std(overall_f1, axis=0)))
    
    print('Negative')
    print('Overall Precision: {} (+/- {}) || Overall Recall: {} (+/- {}) || Overall F1: {} (+/- {})'.format(
                                                                            np.mean(overall_negative_precision), np.std(overall_negative_precision, axis=0),
                                                                           np.mean(overall_negative_recall), np.std(overall_negative_recall, axis=0),
                                                                            np.mean(overall_negative_f1), np.std(overall_negative_f1, axis=0)))
    
def get_cross_validation_two_class(model, X_columns_drop, scale=False):
    """Get cross validated confusion matrix for two class problem"""
    
    # positive scores
    overall_precision = []
    overall_recall = []
    overall_f1 = []
    
    # negative scores
    overall_negative_precision = []
    overall_negative_recall = []
    overall_negative_f1 = []
    
    # loop through all folds
    
    for x in range(5):
        
        # load data
        train = pd.read_csv('/path/to/9_FINAL/data/machine_learning/two_class/one-hot-ecoding/train/cross_validation/training/training_'+str(x)+'.csv',sep=";")
        # debug
        # train_id = train['id']
        del train['Unnamed: 0']
        del train['Unnamed: 0.1']
        y_train = train['label']
        del train['label']
        X_train = train.drop(X_columns_drop, axis=1)
        
        validation = pd.read_csv('/path/to/9_FINAL/data/machine_learning/two_class/one-hot-ecoding/train/cross_validation/validation/val_fold_'+str(x)+'.csv',sep=";")
        # print(validation[validation.id.isin(train_id)])
        del validation['Unnamed: 0']
        y_validation = validation['label']
        del validation['label']
        X_validation = validation.drop(X_columns_drop, axis=1)
        
        # scale if necesary
        if scale:
            min_max_scaler = preprocessing.MinMaxScaler()
            X_train = pd.DataFrame(min_max_scaler.fit_transform(X_train))
            X_validation = pd.DataFrame(min_max_scaler.fit_transform(X_validation))
        
        # fit model
        model.fit(X_train, y_train)
        
        # predicted score
        y_predict = model.predict(X_validation)
        f1 = f1_score(y_validation, y_predict)
        precision = precision_score(y_validation, y_predict)
        recall = recall_score(y_validation, y_predict)
        conf_matrix = confusion_matrix(y_validation, y_predict)
        print(conf_matrix)
        print('Positive')
        print('Precision: {} || Recall: {} || F1: {}'.format(precision, recall, f1))
        
        overall_precision.append(precision)
        overall_recall.append(recall)
        overall_f1.append(f1)
        
        # for negative class (noise), flip labels
        
        precision = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[1,0])
        recall = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[0,1])
        f1 = (2*precision*recall)/(precision+recall)
        
        print('Negative')
        print('Precision: {} || Recall: {} || F1: {}'.format(precision, recall, f1))
        
        overall_negative_precision.append(precision)
        overall_negative_recall.append(recall)
        overall_negative_f1.append(f1)
    
    print('---------------------------')
    print('Positive')
    print('Overall Precision: {} (+/- {}) || Overall Recall: {} (+/- {}) || Overall F1: {} (+/- {})'.format(
                                                                            np.mean(overall_precision), np.std(overall_precision, axis=0),
                                                                           np.mean(overall_recall), np.std(overall_recall, axis=0),
                                                                            np.mean(overall_f1), np.std(overall_f1, axis=0)))
    
    print('Negative')
    print('Overall Precision: {} (+/- {}) || Overall Recall: {} (+/- {}) || Overall F1: {} (+/- {})'.format(
                                                                            np.mean(overall_negative_precision), np.std(overall_negative_precision, axis=0),
                                                                           np.mean(overall_negative_recall), np.std(overall_negative_recall, axis=0),
                                                                            np.mean(overall_negative_f1), np.std(overall_negative_f1, axis=0)))

def get_cross_validation_two_class_normal_distribution(model, X_columns_drop, scale=False):
    """Get cross validated confusion matrix for two class problem"""
    
    # positive scores
    overall_precision = []
    overall_recall = []
    overall_f1 = []
    
    # negative scores
    overall_negative_precision = []
    overall_negative_recall = []
    overall_negative_f1 = []
    
    # loop through all folds
    
    for x in range(5):
        
        # load data
        train = pd.read_csv('/path/to/9_FINAL/data/machine_learning/two_class/one-hot-ecoding/train/cross_validation/training_normal_distribution/training_'+str(x)+'.csv',sep=";")
        # debug
        # train_id = train['id']
        del train['Unnamed: 0']
        y_train = train['label']
        del train['label']
        X_train = train.drop(X_columns_drop, axis=1)
        
        validation = pd.read_csv('/path/to/9_FINAL/data/machine_learning/two_class/one-hot-ecoding/train/cross_validation/validation/val_fold_'+str(x)+'.csv',sep=";")
        # print(validation[validation.id.isin(train_id)])
        del validation['Unnamed: 0']
        y_validation = validation['label']
        del validation['label']
        X_validation = validation.drop(X_columns_drop, axis=1)
        
        # scale if necesary
        if scale:
            min_max_scaler = preprocessing.MinMaxScaler()
            X_train = pd.DataFrame(min_max_scaler.fit_transform(X_train))
            X_validation = pd.DataFrame(min_max_scaler.fit_transform(X_validation))
        
        # fit model
        model.fit(X_train, y_train)
        
        # predicted score
        y_predict = model.predict(X_validation)
        f1 = f1_score(y_validation, y_predict)
        precision = precision_score(y_validation, y_predict)
        recall = recall_score(y_validation, y_predict)
        conf_matrix = confusion_matrix(y_validation, y_predict)
        print(conf_matrix)
        print('Positive')
        print('Precision: {} || Recall: {} || F1: {}'.format(precision, recall, f1))
        
        overall_precision.append(precision)
        overall_recall.append(recall)
        overall_f1.append(f1)
        
        # for negative class (noise), flip labels
        
        precision = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[1,0])
        recall = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[0,1])
        f1 = (2*precision*recall)/(precision+recall)
        
        print('Negative')
        print('Precision: {} || Recall: {} || F1: {}'.format(precision, recall, f1))
        
        overall_negative_precision.append(precision)
        overall_negative_recall.append(recall)
        overall_negative_f1.append(f1)
    
    print('---------------------------')
    print('Positive')
    print('Overall Precision: {} (+/- {}) || Overall Recall: {} (+/- {}) || Overall F1: {} (+/- {})'.format(
                                                                            np.mean(overall_precision), np.std(overall_precision, axis=0),
                                                                           np.mean(overall_recall), np.std(overall_recall, axis=0),
                                                                            np.mean(overall_f1), np.std(overall_f1, axis=0)))
    
    print('Negative')
    print('Overall Precision: {} (+/- {}) || Overall Recall: {} (+/- {}) || Overall F1: {} (+/- {})'.format(
                                                                            np.mean(overall_negative_precision), np.std(overall_negative_precision, axis=0),
                                                                           np.mean(overall_negative_recall), np.std(overall_negative_recall, axis=0),
                                                                            np.mean(overall_negative_f1), np.std(overall_negative_f1, axis=0)))

    
    
def get_count_cross_validation_two_class(model, X_columns_drop, scale=False):
    """Get cross validated confusion matrix for two class problem and count based features"""
    
    # positive scores
    overall_precision = []
    overall_recall = []
    overall_f1 = []
    
    # negative scores
    overall_negative_precision = []
    overall_negative_recall = []
    overall_negative_f1 = []
    
    
    
    # loop through all folds    
    for x in range(5):
        
        # load data
        train = pd.read_csv('/path/to/9_FINAL/data/machine_learning/count_based/cross_validation/training_downsampled/training_'+str(x)+'.csv',sep=";")
        # debug
        # train_id = train['id']
        del train['Unnamed: 0']
        y_train = train['label']
        del train['label']
        X_train = train.drop(X_columns_drop, axis=1)
        
        validation = pd.read_csv('/path/to/9_FINAL/data/machine_learning/count_based/cross_validation/validation/validation_'+str(x)+'.csv',sep=";")
        # print(validation[validation.id.isin(train_id)])
        del validation['Unnamed: 0']
        y_validation = validation['label']
        del validation['label']
        X_validation = validation.drop(X_columns_drop, axis=1)
        
        # scale if necesary
        if scale:
            min_max_scaler = preprocessing.MinMaxScaler()
            X_train = pd.DataFrame(min_max_scaler.fit_transform(X_train))
            X_validation = pd.DataFrame(min_max_scaler.fit_transform(X_validation))
        
        # fit model
        model.fit(X_train, y_train)
        
        # predicted score
        y_predict = model.predict(X_validation)
        f1 = f1_score(y_validation, y_predict)
        precision = precision_score(y_validation, y_predict)
        recall = recall_score(y_validation, y_predict)
        conf_matrix = confusion_matrix(y_validation, y_predict)
        print(conf_matrix)
        print('Positive')
        print('Precision: {} || Recall: {} || F1: {}'.format(precision, recall, f1))
        
        overall_precision.append(precision)
        overall_recall.append(recall)
        overall_f1.append(f1)
        
        # for negative class (noise), flip labels
        
        precision = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[1,0])
        recall = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[0,1])
        f1 = (2*precision*recall)/(precision+recall)
        
        print('Negative')
        print('Precision: {} || Recall: {} || F1: {}'.format(precision, recall, f1))
        
        overall_negative_precision.append(precision)
        overall_negative_recall.append(recall)
        overall_negative_f1.append(f1)
    
    print('---------------------------')
    print('Positive')
    print('Overall Precision: {} (+/- {}) || Overall Recall: {} (+/- {}) || Overall F1: {} (+/- {})'.format(
                                                                            np.mean(overall_precision), np.std(overall_precision, axis=0),
                                                                           np.mean(overall_recall), np.std(overall_recall, axis=0),
                                                                            np.mean(overall_f1), np.std(overall_f1, axis=0)))
    
    print('Negative')
    print('Overall Precision: {} (+/- {}) || Overall Recall: {} (+/- {}) || Overall F1: {} (+/- {})'.format(
                                                                            np.mean(overall_negative_precision), np.std(overall_negative_precision, axis=0),
                                                                           np.mean(overall_negative_recall), np.std(overall_negative_recall, axis=0),
                                                                            np.mean(overall_negative_f1), np.std(overall_negative_f1, axis=0)))

def get_count_cross_validation_two_class_pca(model, X_columns_drop):
    """Get cross validated confusion matrix for two class problem and count based features and scale features with PCA"""
    
    # positive scores
    overall_precision = []
    overall_recall = []
    overall_f1 = []
    
    # negative scores
    overall_negative_precision = []
    overall_negative_recall = []
    overall_negative_f1 = []    
    
    # loop through all folds    
    for x in range(5):
        
        # load data
        train = pd.read_csv('/path/to/9_FINAL/data/machine_learning/count_based/cross_validation/training_downsampled/training_'+str(x)+'.csv',sep=";")
        # debug
        # train_id = train['id']
        del train['Unnamed: 0']
        y_train = train['label']
        del train['label']
        X_train = train.drop(X_columns_drop, axis=1)
        
        # scale training data
        # initialize scaler
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        
        # get validation set
        validation = pd.read_csv('/path/to/9_FINAL/data/machine_learning/count_based/cross_validation/validation/validation_'+str(x)+'.csv',sep=";")
        # print(validation[validation.id.isin(train_id)])
        del validation['Unnamed: 0']
        y_validation = validation['label']
        del validation['label']
        X_validation = validation.drop(X_columns_drop, axis=1)
        X_validation = scaler.transform(X_validation)
        
        # pca on training data
        pca = PCA(n_components=10, random_state=88) # change ratio
        pca.fit(X_train)
        
        print(pca.explained_variance_ratio_) # debugging
        
        # apply mapping of pca to both training and testing
        X_train = pca.transform(X_train)
        X_validation = pca.transform(X_validation)
        
        # fit model
        model.fit(X_train, y_train)
        
        # predicted score
        y_predict = model.predict(X_validation)
        f1 = f1_score(y_validation, y_predict)
        precision = precision_score(y_validation, y_predict)
        recall = recall_score(y_validation, y_predict)
        conf_matrix = confusion_matrix(y_validation, y_predict)
        print(conf_matrix)
        print('Positive')
        print('Precision: {} || Recall: {} || F1: {}'.format(precision, recall, f1))
        
        overall_precision.append(precision)
        overall_recall.append(recall)
        overall_f1.append(f1)
        
        # for negative class (noise), flip labels
        
        precision = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[1,0])
        recall = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[0,1])
        f1 = (2*precision*recall)/(precision+recall)
        
        print('Negative')
        print('Precision: {} || Recall: {} || F1: {}'.format(precision, recall, f1))
        
        overall_negative_precision.append(precision)
        overall_negative_recall.append(recall)
        overall_negative_f1.append(f1)
    
    print('---------------------------')
    print('Positive')
    print('Overall Precision: {} (+/- {}) || Overall Recall: {} (+/- {}) || Overall F1: {} (+/- {})'.format(
                                                                            np.mean(overall_precision), np.std(overall_precision, axis=0),
                                                                           np.mean(overall_recall), np.std(overall_recall, axis=0),
                                                                            np.mean(overall_f1), np.std(overall_f1, axis=0)))
    
    print('Negative')
    print('Overall Precision: {} (+/- {}) || Overall Recall: {} (+/- {}) || Overall F1: {} (+/- {})'.format(
                                                                            np.mean(overall_negative_precision), np.std(overall_negative_precision, axis=0),
                                                                           np.mean(overall_negative_recall), np.std(overall_negative_recall, axis=0),
                                                                            np.mean(overall_negative_f1), np.std(overall_negative_f1, axis=0)))
    
def get_count_cross_validation_two_class_frequency(model, X_columns_drop, scale=False):
    """Get cross validated confusion matrix for two class problem"""
    
    # positive scores
    overall_precision = []
    overall_recall = []
    overall_f1 = []
    
    # negative scores
    overall_negative_precision = []
    overall_negative_recall = []
    overall_negative_f1 = []
    
    # loop through all folds
    
    for x in range(5):
        
        # load data
        train = pd.read_csv('/path/to/9_FINAL/data/machine_learning/count_based/cross_validation/training/training_'+str(x)+'.csv',sep=";")
        # debug
        train_id = train['id']
        del train['Unnamed: 0']
        y_train = train['label']
        del train['label']
        X_train = train[X_columns_drop]
        
        validation = pd.read_csv('/path/to/9_FINAL/data/machine_learning/count_based/cross_validation/validation/validation_'+str(x)+'.csv',sep=";")
        # print(validation[validation.id.isin(train_id)])
        del validation['Unnamed: 0']
        y_validation = validation['label']
        del validation['label']
        X_validation = validation[X_columns_drop]
        
        # scale if necesary
        if scale:
            min_max_scaler = preprocessing.MinMaxScaler()
            X_train = pd.DataFrame(min_max_scaler.fit_transform(X_train))
            X_validation = pd.DataFrame(min_max_scaler.fit_transform(X_validation))
        
        # fit model
        model.fit(X_train, y_train)
        
        # predicted score
        y_predict = model.predict(X_validation)
        f1 = f1_score(y_validation, y_predict)
        precision = precision_score(y_validation, y_predict)
        recall = recall_score(y_validation, y_predict)
        conf_matrix = confusion_matrix(y_validation, y_predict)
        print(conf_matrix)
        print('Positive')
        print('Precision: {} || Recall: {} || F1: {}'.format(precision, recall, f1))
        
        overall_precision.append(precision)
        overall_recall.append(recall)
        overall_f1.append(f1)
        
        # for negative class (noise), flip labels
        
        precision = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[1,0])
        recall = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[0,1])
        f1 = (2*precision*recall)/(precision+recall)
        
        print('Negative')
        print('Precision: {} || Recall: {} || F1: {}'.format(precision, recall, f1))
        
        overall_negative_precision.append(precision)
        overall_negative_recall.append(recall)
        overall_negative_f1.append(f1)
    
    print('---------------------------')
    print('Positive')
    print('Overall Precision: {} (+/- {}) || Overall Recall: {} (+/- {}) || Overall F1: {} (+/- {})'.format(
                                                                            np.mean(overall_precision), np.std(overall_precision, axis=0),
                                                                           np.mean(overall_recall), np.std(overall_recall, axis=0),
                                                                            np.mean(overall_f1), np.std(overall_f1, axis=0)))
    
    print('Negative')
    print('Overall Precision: {} (+/- {}) || Overall Recall: {} (+/- {}) || Overall F1: {} (+/- {})'.format(
                                                                            np.mean(overall_negative_precision), np.std(overall_negative_precision, axis=0),
                                                                           np.mean(overall_negative_recall), np.std(overall_negative_recall, axis=0),
                                                                            np.mean(overall_negative_f1), np.std(overall_negative_f1, axis=0)))

    
def get_cross_validation_two_class_frequency(model, X_columns_drop, scale=False):
    """Get cross validated confusion matrix for two class problem"""
    
    # positive scores
    overall_precision = []
    overall_recall = []
    overall_f1 = []
    
    # negative scores
    overall_negative_precision = []
    overall_negative_recall = []
    overall_negative_f1 = []
    
    # loop through all folds
    
    for x in range(5):
        
        # load data
        train = pd.read_csv('/path/to/9_FINAL/data/machine_learning/two_class/one-hot-ecoding/train/cross_validation/training/training_'+str(x)+'.csv',sep=";")
        # debug
        train_id = train['id']
        del train['Unnamed: 0']
        del train['Unnamed: 0.1']
        y_train = train['label']
        del train['label']
        X_train = train[X_columns_drop]
        
        validation = pd.read_csv('/path/to/9_FINAL/data/machine_learning/two_class/one-hot-ecoding/train/cross_validation/validation/val_fold_'+str(x)+'.csv',sep=";")
        # print(validation[validation.id.isin(train_id)])
        del validation['Unnamed: 0']
        y_validation = validation['label']
        del validation['label']
        X_validation = validation[X_columns_drop]
        
        # scale if necesary
        if scale:
            min_max_scaler = preprocessing.MinMaxScaler()
            X_train = pd.DataFrame(min_max_scaler.fit_transform(X_train))
            X_validation = pd.DataFrame(min_max_scaler.fit_transform(X_validation))
        
        # fit model
        model.fit(X_train, y_train)
        
        # predicted score
        y_predict = model.predict(X_validation)
        f1 = f1_score(y_validation, y_predict)
        precision = precision_score(y_validation, y_predict)
        recall = recall_score(y_validation, y_predict)
        conf_matrix = confusion_matrix(y_validation, y_predict)
        print(conf_matrix)
        print('Positive')
        print('Precision: {} || Recall: {} || F1: {}'.format(precision, recall, f1))
        
        overall_precision.append(precision)
        overall_recall.append(recall)
        overall_f1.append(f1)
        
        # for negative class (noise), flip labels
        
        precision = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[1,0])
        recall = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[0,1])
        f1 = (2*precision*recall)/(precision+recall)
        
        print('Negative')
        print('Precision: {} || Recall: {} || F1: {}'.format(precision, recall, f1))
        
        overall_negative_precision.append(precision)
        overall_negative_recall.append(recall)
        overall_negative_f1.append(f1)
    
    print('---------------------------')
    print('Positive')
    print('Overall Precision: {} (+/- {}) || Overall Recall: {} (+/- {}) || Overall F1: {} (+/- {})'.format(
                                                                            np.mean(overall_precision), np.std(overall_precision, axis=0),
                                                                           np.mean(overall_recall), np.std(overall_recall, axis=0),
                                                                            np.mean(overall_f1), np.std(overall_f1, axis=0)))
    
    print('Negative')
    print('Overall Precision: {} (+/- {}) || Overall Recall: {} (+/- {}) || Overall F1: {} (+/- {})'.format(
                                                                            np.mean(overall_negative_precision), np.std(overall_negative_precision, axis=0),
                                                                           np.mean(overall_negative_recall), np.std(overall_negative_recall, axis=0),
                                                                            np.mean(overall_negative_f1), np.std(overall_negative_f1, axis=0)))
    
def get_cross_validation_two_class_frequency_normal_distribution(model, X_columns_drop, scale=False):
    """Get cross validated confusion matrix for two class problem"""
    
    # positive scores
    overall_precision = []
    overall_recall = []
    overall_f1 = []
    
    # negative scores
    overall_negative_precision = []
    overall_negative_recall = []
    overall_negative_f1 = []
    
    # loop through all folds
    
    for x in range(5):
        
        # load data
        train = pd.read_csv('/path/to/9_FINAL/data/machine_learning/two_class/one-hot-ecoding/train/cross_validation/training_normal_distribution/training_'+str(x)+'.csv',sep=";")
        
        # debug
        train_id = train['id']
        del train['Unnamed: 0']
        y_train = train['label']
        del train['label']
        X_train = train[X_columns_drop]
        
        validation = pd.read_csv('/path/to/9_FINAL/data/machine_learning/two_class/one-hot-ecoding/train/cross_validation/validation/val_fold_'+str(x)+'.csv',sep=";")
        # print(validation[validation.id.isin(train_id)])
        del validation['Unnamed: 0']
        y_validation = validation['label']
        del validation['label']
        X_validation = validation[X_columns_drop]
        
        # scale if necesary
        if scale:
            min_max_scaler = preprocessing.MinMaxScaler()
            X_train = pd.DataFrame(min_max_scaler.fit_transform(X_train))
            X_validation = pd.DataFrame(min_max_scaler.fit_transform(X_validation))
        
        # fit model
        model.fit(X_train, y_train)
        
        # predicted score
        y_predict = model.predict(X_validation)
        f1 = f1_score(y_validation, y_predict)
        precision = precision_score(y_validation, y_predict)
        recall = recall_score(y_validation, y_predict)
        conf_matrix = confusion_matrix(y_validation, y_predict)
        print(conf_matrix)
        print('Positive')
        print('Precision: {} || Recall: {} || F1: {}'.format(precision, recall, f1))
        
        overall_precision.append(precision)
        overall_recall.append(recall)
        overall_f1.append(f1)
        
        # for negative class (noise), flip labels
        
        precision = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[1,0])
        recall = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[0,1])
        f1 = (2*precision*recall)/(precision+recall)
        
        print('Negative')
        print('Precision: {} || Recall: {} || F1: {}'.format(precision, recall, f1))
        
        overall_negative_precision.append(precision)
        overall_negative_recall.append(recall)
        overall_negative_f1.append(f1)
    
    print('---------------------------')
    print('Positive')
    print('Overall Precision: {} (+/- {}) || Overall Recall: {} (+/- {}) || Overall F1: {} (+/- {})'.format(
                                                                            np.mean(overall_precision), np.std(overall_precision, axis=0),
                                                                           np.mean(overall_recall), np.std(overall_recall, axis=0),
                                                                            np.mean(overall_f1), np.std(overall_f1, axis=0)))
    
    print('Negative')
    print('Overall Precision: {} (+/- {}) || Overall Recall: {} (+/- {}) || Overall F1: {} (+/- {})'.format(
                                                                            np.mean(overall_negative_precision), np.std(overall_negative_precision, axis=0),
                                                                           np.mean(overall_negative_recall), np.std(overall_negative_recall, axis=0),
                                                                            np.mean(overall_negative_f1), np.std(overall_negative_f1, axis=0)))
    
def get_performance_on_goldstandard_normal_distribution(model, golddata, y_gold,  X_columns_drop, scale=False):
    '''Get fit model on first fold and apply it on the goldstandard'''
    
    # load first fold and prepare data
    train = pd.read_csv('/path/to/9_FINAL/data/machine_learning/two_class/one-hot-ecoding/train/cross_validation/training_normal_distribution/training_0.csv',sep=";")
    del train['Unnamed: 0']
    y_train = train['label']
    del train['label']
    X_train = train.drop(X_columns_drop, axis=1)
    
    # scale if necessary
    if scale:
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train = pd.DataFrame(min_max_scaler.fit_transform(X_train))
        golddata = pd.DataFrame(min_max_scaler.fit_transform(golddata))
            
    # get model
    model.fit(X_train, y_train)
    
    # get performance metrics
    y_predict = model.predict(golddata)
    f1 = f1_score(y_gold, y_predict)
    precision = precision_score(y_gold, y_predict)
    recall = recall_score(y_gold, y_predict)
    conf_matrix = confusion_matrix(y_gold, y_predict)
    print(conf_matrix)
    print('Positive')
    print('Precision: {} || Recall: {} || F1: {}'.format(precision, recall, f1))
    
    # for negative class (noise), flip labels
        
    precision = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[1,0])
    recall = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[0,1])
    f1 = (2*precision*recall)/(precision+recall)
        
    print('Negative')
    print('Precision: {} || Recall: {} || F1: {}'.format(precision, recall, f1))
    
def get_performance_on_goldstandard(model, golddata, y_gold,  X_columns_drop, scale=False):
    '''Get fit model on first fold and apply it on the goldstandard'''
    
    # load first fold and prepare data
    train = pd.read_csv('/path/to/9_FINAL/data/machine_learning/two_class/one-hot-ecoding/train/cross_validation/training/training_0.csv',sep=";")
    del train['Unnamed: 0']
    del train['Unnamed: 0.1']
    y_train = train['label']
    del train['label']
    X_train = train.drop(X_columns_drop, axis=1)
    
    # scale if necessary
    if scale:
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train = pd.DataFrame(min_max_scaler.fit_transform(X_train))
        golddata = pd.DataFrame(min_max_scaler.fit_transform(golddata))
            
    # get model
    model.fit(X_train, y_train)
    
    # get performance metrics
    y_predict = model.predict(golddata)
    f1 = f1_score(y_gold, y_predict)
    precision = precision_score(y_gold, y_predict)
    recall = recall_score(y_gold, y_predict)
    conf_matrix = confusion_matrix(y_gold, y_predict)
    print(conf_matrix)
    print('Positive')
    print('Precision: {} || Recall: {} || F1: {}'.format(precision, recall, f1))
    
    # for negative class (noise), flip labels
        
    precision = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[1,0])
    recall = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[0,1])
    f1 = (2*precision*recall)/(precision+recall)
        
    print('Negative')
    print('Precision: {} || Recall: {} || F1: {}'.format(precision, recall, f1))
    
def get_count_performance_on_goldstandard(model, golddata, y_gold,  X_columns_drop, scale=False):
    '''Get fit model on first fold and apply it on the goldstandard'''
    
    # load first fold and prepare data
    train = pd.read_csv('/path/to/9_FINAL/data/machine_learning/count_based/cross_validation/training/training_0.csv',sep=";")
    del train['Unnamed: 0']
    y_train = train['label']
    del train['label']
    X_train = train.drop(X_columns_drop, axis=1)
    
    # scale if necessary
    if scale:
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train = pd.DataFrame(min_max_scaler.fit_transform(X_train))
        golddata = pd.DataFrame(min_max_scaler.fit_transform(golddata))
            
    # get model
    model.fit(X_train, y_train)
    
    # get performance metrics
    y_predict = model.predict(golddata)
    f1 = f1_score(y_gold, y_predict)
    precision = precision_score(y_gold, y_predict)
    recall = recall_score(y_gold, y_predict)
    conf_matrix = confusion_matrix(y_gold, y_predict)
    print(conf_matrix)
    print('Positive')
    print('Precision: {} || Recall: {} || F1: {}'.format(precision, recall, f1))
    
    # for negative class (noise), flip labels
        
    precision = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[1,0])
    recall = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[0,1])
    f1 = (2*precision*recall)/(precision+recall)
        
    print('Negative')
    print('Precision: {} || Recall: {} || F1: {}'.format(precision, recall, f1))