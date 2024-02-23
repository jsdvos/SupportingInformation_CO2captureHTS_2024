import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import KFold, cross_validate
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor


def get_data():
    features = pd.read_csv('../data/input/features.csv', sep = ';')
    features['mass'] = features['rho']*features['vol']
    for feature in ['asa', 'av', 'nasa', 'nav']:
        features[feature + '_m'] = features[feature]/features['mass']
        features[feature + '_v'] = features[feature]/features['vol']
    rac_features = ['prod_I_0', 'prod_I_1', 'prod_I_2', 'prod_I_3', 'prod_T_0', 'prod_T_1', 'prod_T_2', 'prod_T_3', 'prod_X_0', 'prod_X_1', 'prod_X_2', 'prod_X_3', 'prod_S_0', 'prod_S_1', 'prod_S_2', 'prod_S_3', 'prod_Z_0', 'prod_Z_1', 'prod_Z_2', 'prod_Z_3', 'prod_a_0', 'prod_a_1', 'prod_a_2', 'prod_a_3', 'diff_I_0', 'diff_I_1', 'diff_I_2', 'diff_I_3', 'diff_T_0', 'diff_T_1', 'diff_T_2', 'diff_T_3', 'diff_X_0', 'diff_X_1', 'diff_X_2', 'diff_X_3', 'diff_S_0', 'diff_S_1', 'diff_S_2', 'diff_S_3', 'diff_Z_0', 'diff_Z_1', 'diff_Z_2', 'diff_Z_3', 'diff_a_0', 'diff_a_1', 'diff_a_2', 'diff_a_3']
    for feature in rac_features:
        features['linkresc_' + feature] = np.where(features['link_N_start'] != 0, features['link_' + feature]*features['link_N_start']/features['link_N_part'], 0.0)

    results = pd.read_csv('../data/input/results.csv', sep = ';')
    results['D_co2 [mg/g]'] = results['co2_ads [mg/g]'] - results['co2_des [mg/g]']
    m_n2_co2 = 0.6365307490428204
    results['S_co2/n2'] = results['co2_ads [mg/g]']/results['n2_ads [mg/g]']*m_n2_co2

    data = features.merge(results, how = 'outer', on = 'struct')
    return data

def preprocess(data, non_accessible = True, rescale = True):
    # Drop struct, database, q_st
    data.drop(['database', 'q_st_co2_des [kJ/mol]', 'q_st_co2_ads [kJ/mol]', 'q_st_n2_ads [kJ/mol]'], axis = 1, inplace = True)
    # Drop other non-informative labels that only contain extensive properties
    data.drop(['vol', 'mass', 'asa', 'av', 'nasa', 'nav'], axis = 1, inplace = True)
    # Only retain (not-)rescaled linker RACs
    rac_features = ['prod_I_0', 'prod_I_1', 'prod_I_2', 'prod_I_3', 'prod_T_0', 'prod_T_1', 'prod_T_2', 'prod_T_3', 'prod_X_0', 'prod_X_1', 'prod_X_2', 'prod_X_3', 'prod_S_0', 'prod_S_1', 'prod_S_2', 'prod_S_3', 'prod_Z_0', 'prod_Z_1', 'prod_Z_2', 'prod_Z_3', 'prod_a_0', 'prod_a_1', 'prod_a_2', 'prod_a_3', 'diff_I_0', 'diff_I_1', 'diff_I_2', 'diff_I_3', 'diff_T_0', 'diff_T_1', 'diff_T_2', 'diff_T_3', 'diff_X_0', 'diff_X_1', 'diff_X_2', 'diff_X_3', 'diff_S_0', 'diff_S_1', 'diff_S_2', 'diff_S_3', 'diff_Z_0', 'diff_Z_1', 'diff_Z_2', 'diff_Z_3', 'diff_a_0', 'diff_a_1', 'diff_a_2', 'diff_a_3']
    data.drop(['conn_' + feature for feature in rac_features], axis = 1, inplace = True)
    data.drop([x for x in data.columns if x.endswith('_N_start')], axis = 1, inplace = True)
    data.drop([x for x in data.columns if x.endswith('_N_part')], axis = 1, inplace = True)
    if rescale:
        data.drop(['link_' + feature for feature in rac_features], axis = 1, inplace = True)
    else:
        data.drop(['linkresc_' + feature for feature in rac_features], axis = 1, inplace = True)
    if not non_accessible:
        # Drop nasa, nav
        data.drop(['nasa_m', 'nav_m', 'nasa_v', 'nav_v'], axis = 1, inplace = True)

    # Drop labels with STD=0
    features = data.columns
    features = features.drop('struct')
    start = len(features)
    features = [feature for feature in features if not np.std(data[feature]) < 1e-9]
    stop = len(features)
    print('Removed {} features with STD=0 ({} -> {})'.format(start - stop, start, stop))
    print([x for x in features])
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
    feats = x.columns
    return struct.values, x.values, y.values, feats

def normalize(x_train, x_test, x_exp, y_train, y_test):
    x_scaler = StandardScaler().fit(x_train)
    x_train = x_scaler.transform(x_train)
    x_test = x_scaler.transform(x_test)
    x_exp = x_scaler.transform(x_exp)
    y_scaler = StandardScaler().fit(y_train)
    y_train = y_scaler.transform(y_train)
    y_test = y_scaler.transform(y_test)
    return x_train, x_test, x_exp, y_train, y_test, y_scaler

if __name__ == '__main__':
    # Load data
    data = get_data()
    data = preprocess(data)
    struct_train, x_train, y_train, feats_train = split_data(data, 'train')
    struct_test, x_test, y_test, feats_test = split_data(data, 'test')
    struct_all, x_all, y_all, feats_all = split_data(data, 'all')
    assert (feats_train == feats_test).all() and (feats_train == feats_all).all()
    x_train, x_test, x_all, y_train, y_test, y_scaler = normalize(x_train, x_test, x_all,
            y_train, y_test)
    result = {
            'x_train': x_train,
            'x_test': x_test,
            'x_all': x_all,
            'y_train': y_train,
            'y_test': y_test,
            'y_scaler': y_scaler
            }

    # Feature selection
    kf10 = KFold(n_splits = 10, shuffle = True)
    estimators = [
            SVR(**{'C': 5.0, 'epsilon': 0.0, 'gamma': 0.008, 'kernel': 'rbf'}),
            GradientBoostingRegressor(**{'learning_rate': 0.01, 'loss': 'absolute_error', 'max_depth': 20, 'min_samples_split': 3, 'n_estimators': 1000}),
            KernelRidge(**{'alpha': 0.001, 'degree': 2, 'gamma': 0.01, 'kernel': 'polynomial'}),
            SVR(**{'C': 5.0, 'epsilon': 0.01, 'gamma': 0.008, 'kernel': 'rbf'}),
            GradientBoostingRegressor(**{'learning_rate': 0.01, 'loss': 'absolute_error', 'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 1000})
            ]
    for i in range(5):
        plt.figure()
        means = []
        stds = []
        x = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 113])
        for k in x:
            select = SelectKBest(f_regression, k = k)
            select.fit(x_train, y_train[:, i])
            if k == 5:
                pvalues = select.pvalues_
            x_select = select.transform(x_train)
            scores = cross_validate(
                    estimator = estimators[i],
                    X = x_select,
                    y = y_train[:, i],
                    cv = kf10,
                    scoring = 'neg_mean_absolute_error',
                    n_jobs = -1)
            means.append(-np.mean(scores['test_score']))
            stds.append(np.std(scores['test_score']))
        plt.errorbar(x, means, yerr = stds)
        plt.xlabel('Number of features selected')
        plt.ylabel('Mean absolute error [-]')
        plt.savefig('../figs/feature_reduction/SelectKBest_T{}.pdf'.format(i), bbox_inches = 'tight')
        plt.close()

