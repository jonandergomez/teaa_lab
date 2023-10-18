import os
import sys
import time
import math
import numpy
import gzip
import pickle
import tempfile

from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier

from utils_for_results import save_results

# --------------------------------------------------------------------------------
def generate_y_true(time_to_seizure):
    tts = time_to_seizure
    y_true = numpy.zeros(len(tts), dtype = int)
    y_true[tts >   2     ] = 1
    y_true[tts >  10 * 60] = 2
    y_true[tts >  20 * 60] = 3
    y_true[tts <    -10  ] = 4
    y_true[tts < -20 * 60] = 5
    y_true[tts == 0      ] = 0 # redundant, but just in case
    return y_true
# --------------------------------------------------------------------------------
def csv_line_to_patient_tts_label_and_sample(line):
    parts = line.split(';')
    patient = parts[0]
    tts = float(parts[1])
    if do_binary_classification:
        label = 0 if tts == 0 else 1
    else:
        label = 0
        ### BEGIN: Students can change this to check other approaches
        if tts >   2     : label = 1
        if tts >  10 * 60: label = 2
        if tts >  20 * 60: label = 3
        if tts < -10     : label = 4
        if tts < -20 * 60: label = 5
        ### END: Students can change this to check other approaches
    x = numpy.array([float(x) for x in parts[2:]])
    return [patient, tts, label, x]
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
def load_csv_from_uc13(filenames):
    data = []
    for filename in filenames:
        print(f'loading {filename}')
        f = gzip.open(filename, 'rt')
        for line in f:
            data.append(csv_line_to_patient_tts_label_and_sample(line.strip()))
        f.close()
    print(f'loaded {len(data)} samples')
    return data
# --------------------------------------------------------------------------------

if __name__ == "__main__":

    verbose = 0
    base_dir = '.'
    patient = 'chb03'
    data_format = 'pca136'
    do_train = False
    do_classify = False
    do_binary_classification = False

    for i in range(1, len(sys.argv)):
        if   sys.argv[i] == '--patient'                 : patient = sys.argv[i + 1]
        elif sys.argv[i] == '--data-format'             : data_format = sys.argv[i + 1]
        elif sys.argv[i] == '--models-dir'              : models_dir = sys.argv[i + 1]
        elif sys.argv[i] == '--do-binary-classification': do_binary_classification = True
        elif sys.argv[i] == '-v'                        : verbose += 1

    do_z_transform = (data_format == '21x14') # It should be False if PCA was applied

    models_dir  = f'{base_dir}/models/uc13/ensembles'
    log_dir     = f'{base_dir}/logs/uc13/ensembles'
    results_dir = f'{base_dir}/results/uc13/ensembles/{patient}'

    task = 'binary-classification' if do_binary_classification else 'multi-class-classification'

    os.makedirs(log_dir,     exist_ok = True)
    os.makedirs(models_dir,  exist_ok = True)
    os.makedirs(results_dir, exist_ok = True)

    if patient == 'ALL':
        train_filenames = [f'data/21x14/uc13-chb{i:02d}-{data_format}-time-to-seizure.csv.gz' for i in range(1,17)]
        test_filenames  = [f'data/21x14/uc13-chb{i:02d}-{data_format}-time-to-seizure.csv.gz' for i in range(17,25)]
    else:
        train_filenames = [f'data/21x14/uc13-{patient}-{data_format}-time-to-seizure-train.csv.gz']
        test_filenames  = [f'data/21x14/uc13-{patient}-{data_format}-time-to-seizure-test.csv.gz']

    # Loads and repartitions the data
    rdd_train = load_csv_from_uc13(train_filenames)
    rdd_test  = load_csv_from_uc13( test_filenames)

    X_train = numpy.array([x[3] for x in rdd_train])
    y_train = numpy.array([x[2] for x in rdd_train])
    X_test = numpy.array([x[3] for x in rdd_test])
    y_test = numpy.array([x[2] for x in rdd_test])
    # BEGIN: Perform the standard scalation
    if do_z_transform:
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        for i in range(len(rdd_train)):
            rdd_train[i][3] = X_train[i].copy()
        #del X_train
        #
        X_test = scaler.transform(X_test)
        for i in range(len(rdd_test)):
            rdd_test[i][3] = X_test[i].copy()
        #del X_test
    # END: Perform the standard scalation

    if do_binary_classification:
        labels = [0, 1]
    else:
        l1 = [x[2] for x in rdd_train]
        print(numpy.unique(l1))
        l2 = [x[2] for x in rdd_test]
        labels = list(numpy.unique(l1 + l2))

    print('labels', labels)

    
    for n_estimators in [100, 200, 300, 500, 700, 1000]:
        for max_depth in [3, 5, 7, 9, 11]:

            print(f'working with samples {X_train.shape}, {n_estimators} estimators and a max depth equal to {max_depth}')

            rf = RandomForestClassifier(n_estimators = n_estimators, criterion = 'gini', max_depth = max_depth)
            ert = ExtraTreesClassifier(n_estimators = n_estimators, criterion = 'gini', max_depth = max_depth)
            gbt = HistGradientBoostingClassifier(max_iter = n_estimators, loss = 'log_loss', max_depth = max_depth)

            rf.fit(X_train, y_train)
            ert.fit(X_train, y_train)
            gbt.fit(X_train, y_train)

            # Random Forest
            subset = 'train' 
            y_true = y_train
            y_pred = rf.predict(X_train)

            filename_prefix = f'rf-{patient}-{data_format}-{task}-{n_estimators:04d}-{max_depth:03d}'
            save_results(f'{results_dir}/rf/{subset}', filename_prefix, y_true, y_pred, labels = labels)

            subset = 'test' 
            y_true = y_test
            y_pred = rf.predict(X_test)

            save_results(f'{results_dir}/rf/{subset}', filename_prefix, y_true, y_pred, labels = labels)

            # Extremely Randomized Trees
            subset = 'train' 
            y_true = y_train
            y_pred = ert.predict(X_train)

            filename_prefix = f'ert-{patient}-{data_format}-{task}-{n_estimators:04d}-{max_depth:03d}'
            save_results(f'{results_dir}/ert/{subset}', filename_prefix, y_true, y_pred, labels = labels)

            subset = 'test' 
            y_true = y_test
            y_pred = ert.predict(X_test)

            save_results(f'{results_dir}/ert/{subset}', filename_prefix, y_true, y_pred, labels = labels)

            # Gradient Boosted Trees
            subset = 'train' 
            y_true = y_train
            y_pred = gbt.predict(X_train)

            filename_prefix = f'gbt-{patient}-{data_format}-{task}-{n_estimators:04d}-{max_depth:03d}'
            save_results(f'{results_dir}/gbt/{subset}', filename_prefix, y_true, y_pred, labels = labels)

            subset = 'test' 
            y_true = y_test
            y_pred = gbt.predict(X_test)

            save_results(f'{results_dir}/gbt/{subset}', filename_prefix, y_true, y_pred, labels = labels)
