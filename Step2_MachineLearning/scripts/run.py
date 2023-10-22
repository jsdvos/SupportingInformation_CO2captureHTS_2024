import os
import sys
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_validate, KFold

import shap

np.random.seed(2202151891)

############################################################################
###############################     PREPROCESSING    #######################
############################################################################

def get_data():
    features = pd.read_csv('../data/input/features.csv', sep = ';')
    features['mass'] = features['rho']*features['vol']
    for feature in ['asa', 'av', 'nasa', 'nav']:
        features[feature + '_m'] = features[feature]/features['mass']
        features[feature + '_v'] = features[feature]/features['vol']
    rac_features = ['prod_I_0', 'prod_I_1', 'prod_I_2', 'prod_I_3', 'prod_T_0', 'prod_T_1', 'prod_T_2', 'prod_T_3', 'prod_X_0', 'prod_X_1', 'prod_X_2', 'prod_X_3', 'prod_S_0', 'prod_S_1', 'prod_S_2', 'prod_S_3', 'prod_Z_0', 'prod_Z_1', 'prod_Z_2', 'prod_Z_3', 'prod_a_0', 'prod_a_1', 'prod_a_2', 'prod_a_3', 'diff_I_0', 'diff_I_1', 'diff_I_2', 'diff_I_3', 'diff_T_0', 'diff_T_1', 'diff_T_2', 'diff_T_3', 'diff_X_0', 'diff_X_1', 'diff_X_2', 'diff_X_3', 'diff_S_0', 'diff_S_1', 'diff_S_2', 'diff_S_3', 'diff_Z_0', 'diff_Z_1', 'diff_Z_2', 'diff_Z_3', 'diff_a_0', 'diff_a_1', 'diff_a_2', 'diff_a_3']
    for feature in rac_features:
        features['linkresc_' + feature] = np.where(features['link_N_start'] != 0, features['link_' + feature]*features['link_N_start']/features['link_N_part'], 0.0)
    
    results = pd.read_csv('../data/feat/results.csv', sep = ';')
    results['D_co2 [mg/g]'] = results['co2_ads [mg/g]'] - results['co2_des [mg/g]'] # Unit is arbitrary, since normalization is performed
    m_n2_co2 = 0.6365307490428204 # M_n2/M_co2
    results['S_co2/n2'] = results['co2_ads [mg/g]']/results['n2_ads [mg/g]']*m_n2_co2 # Conversion to ratio of number of molecules, not ratio of molecules mass

    data = features.merge(results, how = 'outer', on = 'struct')

    return data

def preprocess(data, non_accessible = True, rescale = True):
    # Drop struct, database, q_st
    data.drop(['database', 'q_st_co2_des [kJ/mol]', 'q_st_co2_ads [kJ/mol]', 'q_st_n2_ads [kJ/mol]'], axis = 1, inplace = True)
    # Drop other non-informative labels that only contain extensive properties
    data.drop(['vol', 'mass', 'asa', 'av', 'nasa', 'nav'], axis = 1, inplace = True)
    # Only retain rescaled linker RACs
    rac_features = ['prod_I_0', 'prod_I_1', 'prod_I_2', 'prod_I_3', 'prod_T_0', 'prod_T_1', 'prod_T_2', 'prod_T_3', 'prod_X_0', 'prod_X_1', 'prod_X_2', 'prod_X_3', 'prod_S_0', 'prod_S_1', 'prod_S_2', 'prod_S_3', 'prod_Z_0', 'prod_Z_1', 'prod_Z_2', 'prod_Z_3', 'prod_a_0', 'prod_a_1', 'prod_a_2', 'prod_a_3', 'diff_I_0', 'diff_I_1', 'diff_I_2', 'diff_I_3', 'diff_T_0', 'diff_T_1', 'diff_T_2', 'diff_T_3', 'diff_X_0', 'diff_X_1', 'diff_X_2', 'diff_X_3', 'diff_S_0', 'diff_S_1', 'diff_S_2', 'diff_S_3', 'diff_Z_0', 'diff_Z_1', 'diff_Z_2', 'diff_Z_3', 'diff_a_0', 'diff_a_1', 'diff_a_2', 'diff_a_3']
    data.drop(['conn_' + feature for feature in rac_features], axis = 1, inplace = True)
    data.drop([x for x in data.columns if x.endswith('_N_start')], axis = 1, inplace = True)
    data.drop([x for x in data.columns if x.endswith('_N_part')], axis = 1, inplace = True)
    data.drop(['link_' + feature for feature in rac_features], axis = 1, inplace = True)

    # Drop uniform features with STD=0
    features = data.columns
    features = features.drop('struct')
    start = len(features)
    features = [feature for feature in features if not np.std(data[feature]) < 1e-9]
    stop = len(features)
    print('Non-uniform linkage RACs: {}'.format(len([x for x in features if x.startswith('lig')])))
    print('Non-uniform linker RACs: {}'.format(len([x for x in features if x.startswith('link')])))
    print('Non-uniform func RACs: {}'.format(len([x for x in features if x.startswith('func')])))
    data = data[['struct'] + features]
    return data

def split_data(data, set_label = 'all'):
    mask = np.ones(len(data), dtype = bool)
    assert set_label in ['train', 'test', 'exp', 'all']
    if set_label == 'train':
        fns_struct = ['../data/input/structs_train.txt']
    elif set_label == 'test':
        fns_struct = ['../data/input/structs_test.txt']
    elif set_label == 'exp':
        fns_struct = ['../data/input/structs_train.txt',
                        '../data/input/structs_test.txt']
    else:
        fns_struct = []
    structs = []
    for fn in fns_struct:
        with open(fn, 'r') as f:
            for line in f:
                structs.append(line.strip())
    if set_label in ['train', 'test', 'exp']:
        mask_struct = np.array([struct in structs for struct in data['struct']])
        if set_label == 'exp':
            mask_struct = ~mask_struct
        mask = mask & mask_struct
    
    if set_label in ['train', 'test', 'exp']:
        y_labels = ['co2_ads [mg/g]', 'co2_des [mg/g]',
                'n2_ads [mg/g]', 'D_co2 [mg/g]', 'S_co2/n2']
        labels = data[y_labels]
        mask_nan = ~labels.isnull().sum(axis = 1).astype(dtype = bool)
        if set_label == 'exp':
            mask_nan = ~mask_nan
        mask = mask & mask_nan
    
    data_set = data[mask].copy()
    struct = data_set['struct']
    y_labels = ['co2_ads [mg/g]', 'co2_des [mg/g]', 'n2_ads [mg/g]', 'D_co2 [mg/g]', 'S_co2/n2']
    y = data_set[y_labels]
    x = data_set.drop(['struct'] + y_labels, axis = 1)
    assert x.isnull().sum().sum() == 0
    return struct.values, x.values, y.values, x.columns

def normalize(x_train, x_test, x_exp, y_train, y_test):
    x_scaler = StandardScaler().fit(x_train)
    x_train = x_scaler.transform(x_train)
    x_test = x_scaler.transform(x_test)
    x_exp = x_scaler.transform(x_exp)
    y_scaler = StandardScaler().fit(y_train)
    y_train = y_scaler.transform(y_train)
    y_test = y_scaler.transform(y_test)
    return x_train, x_test, x_exp, y_train, y_test, y_scaler

######################################################################################
###############################    MODEL SELECTION    ################################
######################################################################################

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def wmae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred, sample_weight = y_true - min(y_true))

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared = False)

def wrmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared = False, sample_weight = y_true - min(y_true))

def train_model(x_train, y_train, model):
    regressor = {
            'KRR': KernelRidge(),
            'SVR': SVR(),
            'MLP': MLPRegressor(),
            'RF': RandomForestRegressor(),
            'GB': GradientBoostingRegressor()
            }[model.split('_')[0]]
    params = {
            'KRR': {
                    'alpha': [0.00001, 0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 5.0],
                    'kernel': ['linear', 'rbf', 'laplacian'],
                    'gamma': [None, 1.0/125, 1.0/100, 1.0/75, 1.0/50] # Default (None): 1.0/n_features = 1/151
                    },
            'KRR_poly': {
                    'alpha': [0.00001, 0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 5.0],
                    'kernel': ['polynomial'],
                    'gamma': [None, 1.0/125, 1.0/100, 1.0/75, 1.0/50], # Default (None): 1.0/n_features = 1/151
                    'degree': [2, 3, 4, 5, 6]
                    },
            'SVR': {
                    'kernel': ['linear', 'poly', 'rbf'],
                    'C': [0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],
                    'gamma': ['scale', 'auto', 1.0/125, 1.0/100, 1.0/75, 1.0/50],
                    'epsilon': [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
                    },
            'RF': {
                    'n_estimators': [100, 150, 200, 350, 500, 1000],
                    'criterion': ['squared_error', 'friedman_mse'],
                    'max_depth': [None, 3, 4, 5, 20],
                    'min_samples_split': [2, 3, 4], 
                    'ccp_alpha': [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
                    },
            'RF_abs': {
                    'n_estimators': [100, 150, 200, 350],
                    'criterion': ['absolute_error'],
                    'max_depth': [None],
                    'min_samples_split': [2, 3, 4], 
                    'ccp_alpha': [0.0, 0.005, 0.01, 0.015],
                    },
            'MLP': {
                    'hidden_layer_sizes': [(100, ), (100, 100), (100, 100, 100), (100, 10, 100)],
                    'activation': ['logistic', 'relu', 'identity', 'tanh'],
                    'alpha': [0.0001, 0.01, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0],
                    'learning_rate': ['constant', 'invscaling', 'adaptive'],
                    'max_iter': [2000]
                    },
            'GB_sq': {
                    'loss': ['squared_error'],
                    'n_estimators': [50, 100, 200, 300, 500, 750, 1000],
                    'min_samples_split': [2, 3, 4, 5],
                    'max_depth': [None, 3, 4, 5, 20],
                    'learning_rate': [0.00001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
                    },
            'GB_abs': {
                    'loss': ['absolute_error'],
                    'n_estimators': [50, 100, 200, 300, 500, 750, 1000],
                    'min_samples_split': [2, 3, 4, 5],
                    'max_depth': [None, 3, 4, 5, 20],
                    'learning_rate': [0.00001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
                    },
            'GB_hub1': {
                    'loss': ['huber'],
                    'n_estimators': [50, 100, 200, 300, 500, 750, 1000],
                    'min_samples_split': [2],
                    'max_depth': [None, 3, 4, 5, 20],
                    'learning_rate': [0.00001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
                    },
            'GB_hub2': {
                    'loss': ['huber'],
                    'n_estimators': [50, 100, 200, 300, 500, 750, 1000],
                    'min_samples_split': [3],
                    'max_depth': [None, 3, 4, 5, 20],
                    'learning_rate': [0.00001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
                    },
            'GB_hub3': {
                    'loss': ['huber'],
                    'n_estimators': [50, 100, 200, 300, 500, 750, 1000],
                    'min_samples_split': [4],
                    'max_depth': [None, 3, 4, 5, 20],
                    'learning_rate': [0.00001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
                    },
            'GB_hub4': {
                    'loss': ['huber'],
                    'n_estimators': [50, 100, 200, 300, 500, 750, 1000],
                    'min_samples_split': [5],
                    'max_depth': [None, 3, 4, 5, 20],
                    'learning_rate': [0.00001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
                    },
            'GB_qua': {
                    'loss': ['quantile'],
                    'n_estimators': [50, 100, 200, 300, 500, 750, 1000],
                    'min_samples_split': [2, 3, 4, 5],
                    'max_depth': [None, 3, 4, 5, 20],
                    'learning_rate': [0.00001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
                    },
            }[model]
    kf10 = KFold(n_splits = 10, shuffle = True)
    scores = {
            'mae': make_scorer(mae, greater_is_better = False),
            'wmae': make_scorer(wmae, greater_is_better = False),
            'rmse': make_scorer(rmse, greater_is_better = False),
            'wrmse': make_scorer(wrmse, greater_is_better = False),
            }
    grid_search = GridSearchCV(regressor, params,
                            scoring = scores,
                            n_jobs = -1,
                            cv = kf10,
                            refit = False,
                            verbose = 3)
    if model == 'MLP':
        grid_search.fit(x_train, y_train)
    else:
        grid_search.fit(x_train, y_train, sample_weight = y_train - min(y_train))
    return pd.DataFrame(grid_search.cv_results_)

def write_training(df, fn):
    df.to_csv(fn)
    columns = list(df.columns.values)
    labels = ['mean_test_' + x for x in ['mae', 'wmae', 'rmse', 'wrmse']]
    for label in columns:
        if label.startswith('param_'):
            labels.append(label)
    for label in labels[::-1]:
        columns.remove(label)
        columns = [label] + columns
    df.to_csv(fn, columns = columns)


######################################################################################
###############################    MODEL PREDICTION    ###############################
######################################################################################

def test_model(x, y, model):
    ml_model = {
            'KRR0': KernelRidge(**{'alpha': 0.1, 'degree': 3, 'gamma': 0.01, 'kernel': 'polynomial'}),
            'SVR0': SVR(**{'C': 5.0, 'epsilon': 0.0, 'gamma': 0.008, 'kernel': 'rbf'}),
            'MLP0': MLPRegressor(**{'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100), 'learning_rate': 'invscaling', 'max_iter': 2000}),
            'RF0': RandomForestRegressor(**{'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 200}),
            'GB0': GradientBoostingRegressor(**{'learning_rate': 0.15, 'loss': 'squared_error', 'max_depth': 5, 'min_samples_split': 3, 'n_estimators': 1000}),

            'KRR1': KernelRidge(**{'alpha': 5.0, 'degree': 4, 'gamma': 0.008, 'kernel': 'polynomial'}),
            'SVR1': SVR(**{'C': 3.0, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'rbf'}),
            'MLP1': MLPRegressor(**{'activation': 'relu', 'alpha': 0.25, 'hidden_layer_sizes': (100, 100, 100), 'learning_rate': 'adaptive', 'max_iter': 2000}),
            'RF1': RandomForestRegressor(**{'ccp_alpha': 0.0, 'criterion': 'friedman_mse', 'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 1000}),
            'GB1_mae': GradientBoostingRegressor(**{'learning_rate': 0.01, 'loss': 'absolute_error', 'max_depth': 20, 'min_samples_split': 3, 'n_estimators': 1000}),

            'KRR2': KernelRidge(**{'alpha': 0.001, 'degree': 2, 'gamma': 0.01, 'kernel': 'polynomial'}),
            'SVR2': SVR(**{'C': 0.01, 'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'linear'}),
            'MLP2': MLPRegressor(**{'activation': 'identity', 'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100), 'learning_rate': 'constant', 'max_iter': 2000}),
            'RF2': RandomForestRegressor(**{'ccp_alpha': 0.0, 'criterion': 'friedman_mse', 'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 200}),
            'GB2': GradientBoostingRegressor(**{'learning_rate': 0.05, 'loss': 'squared_error', 'max_depth': 5, 'min_samples_split': 3, 'n_estimators': 1000}),

            'KRR3': KernelRidge(**{'alpha': 0.1, 'degree': 3, 'gamma': 0.01, 'kernel': 'polynomial'}),
            'SVR3': SVR(**{'C': 5.0, 'epsilon': 0.01, 'gamma': 0.008, 'kernel': 'rbf'}),
            'MLP3': MLPRegressor(**{'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100), 'learning_rate': 'constant', 'max_iter': 2000}),
            'RF3': RandomForestRegressor(**{'ccp_alpha': 0.0, 'criterion': 'friedman_mse', 'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 1000}),
            'GB3': GradientBoostingRegressor(**{'learning_rate': 0.1, 'loss': 'squared_error', 'max_depth': 5, 'min_samples_split': 2, 'n_estimators': 1000}),

            'KRR4': KernelRidge(**{'alpha': 5.0, 'degree': 3, 'gamma': 0.02, 'kernel': 'polynomial'}),
            'SVR4': SVR(**{'C': 3.0, 'epsilon': 0.0, 'gamma': 0.01, 'kernel': 'rbf'}),
            'MLP4': MLPRegressor(**{'activation': 'logistic', 'alpha': 0.0001, 'hidden_layer_sizes': (100, 100), 'learning_rate': 'adaptive', 'max_iter': 2000}),
            'RF4': RandomForestRegressor(**{'ccp_alpha': 0.0, 'criterion': 'friedman_mse', 'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 1000}),
            'GB4': GradientBoostingRegressor(**{'learning_rate': 0.01, 'loss': 'absolute_error', 'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 1000}),
            }[model]
    if model.startswith('MLP'):
        ml_model.fit(x, y)
    else:
        ml_model.fit(x, y, sample_weight = y - min(y))
    return ml_model

################################################################################################
#########################################    MAIN    ###########################################
################################################################################################

if __name__ == '__main__':
    # Initialize training and test sets
    data = get_data()
    data = preprocess(data)
    struct_train, x_train, y_train, feat_train = split_data(data, 'train')
    struct_test, x_test, y_test, feat_test = split_data(data, 'test')
    struct_all, x_all, y_all, feat_all  = split_data(data, 'all')
    x_train, x_test, x_all, y_train, y_test, y_scaler = normalize(x_train, x_test, x_all,
            y_train, y_test)
    assert (feat_train == feat_test).all()
    assert (feat_train == feat_all).all()
    features = feat_train

    if sys.argv[1] == 'train':
        for algorithm in ['KRR', 'KRR_poly', 'SVR', 'RF', 'RF_abs', 'MLP', 'GB_sq', 'GB_abs', 'GB_hub1', 'GB_hub2', 'GB_hub3', 'GB_hub4', 'GB_qua']:
            for run in [0, 1, 2, 3, 4]:
                df = train_model(x_train, y_train[:, run], algorithm)
                write_training(df, '../data/training/train_{}{}.csv'.format(algorithm, run))

    elif sys.argv[1] == 'predict':
        for algorithm in ['KRR', 'SVR', 'MLP', 'RF', 'GB']:
            for run in [0, 1, 2, 3, 4]:
                # Train model
                model = '{}{}'.format(algorithm, run)
                ml_model = test_model(x_train, y_train[:, run], model)
                # Predict all materials in database
                y_predict = ml_model.predict(x_all)
                with open('../data/predictions/exp_{}.csv'.format(model), 'w') as f:
                    f.write('struct,y_predict\n')
                    for i in range(len(y_predict)):
                        f.write('{},{}\n'.format(struct_all[i], y_predict[i]))
                # Predict all materials in test set
                y_predict = ml_model.predict(x_test)
                with open('../data/predictions/test_{}.csv'.format(model), 'w') as f:
                    f.write('struct,y_test,y_predict\n')
                    for i in range(len(y_predict)):
                        f.write('{},{},{}\n'.format(struct_test[i], y_test[i][run], y_predict[i]))

    elif sys.argv[1] == 'shap':
        for algorithm in ['KRR', 'SVR', 'MLP', 'RF', 'GB']:
            for run in [0, 1, 2, 3, 4]:
                model = '{}{}'.format(algorithm, run)
                ml_model = test_model(x_train, y_train[:, run], model)
                # Fit the SHAP explainer
                seed = int(np.random.rand()*(2**32-1))
                explainer = shap.Explainer(ml_model.predict, x_test, seed = seed)
                # Calculate the SHAP values
                shap_values = explainer(x_test)
                df = pd.DataFrame(shap_values, columns = features)
                df['struct'] = struct_test
                df.to_csv('../data/shap/shap_{}.csv'.format(model))

