#
#
#
#
#
#

"""
Extremely Randomized Trees Classifier Example.
"""

import sys
import os
import argparse
import numpy
import time

from sklearn.ensemble import ExtraTreesClassifier

from utils_for_results import save_results


def main(args):

    # Function to process each CSV line
    def process_csv_line(row):
        # returns a list with patient, label and features
        return row[0], int(label_mapping[int(row[1])]), numpy.array([float(x) for x in row[2:]])

    # Loading and parsing the data file, converting it to a DataFrame.
    if args.usingPCA:
        train_filename = f'{args.dataDir}/uc13-{args.patient}-21x20-train-pca.csv'
        test_filename  = f'{args.dataDir}/uc13-{args.patient}-21x20-test-pca.csv'
    else:
        train_filename = f'{args.dataDir}/uc13-{args.patient}-21x20-train.csv'
        test_filename  = f'{args.dataDir}/uc13-{args.patient}-21x20-test.csv'

    # Preparing the labels according to the number of target classes
    #   2 for binary classification
    #  10 for multi-class classification
    label_mapping = [i for i in range(10)]
    if args.doBinaryClassification:
        label_mapping = [0 for i in range(10)]
        label_mapping[1] = 1
    labels = numpy.unique(label_mapping)

    # Load data
    X_train = list()
    y_train = list()
    X_test = list()
    y_test = list()

    def load_data(filename):
        _X_ = list()
        _y_ = list()
        with open(filename, 'rt') as f:
            for line in f:
                row = process_csv_line(line.strip().split(sep = ';'))
                _X_.append(row[2])
                _y_.append(row[1])
            f.close()
        return numpy.array(_X_), numpy.array(_y_)

    X_train, y_train = load_data(train_filename)
    X_test,  y_test  = load_data(test_filename)


    results_dir = f'{args.baseDir}/{args.resultsDir}'
    models_dir  = f'{args.baseDir}/{args.modelsDir}'
    log_dir     = f'{args.baseDir}/{args.logDir}'
    #os.makedirs(log_dir,     exist_ok = True)
    #os.makedirs(models_dir,  exist_ok = True)
    #os.makedirs(results_dir, exist_ok = True)

    elapsed_time = {'training': 0, 'inference_train': 0, 'inference_test': 0}

    # Creating the ExtraTrees model
    reference_time = time.time()
    ert = ExtraTreesClassifier(n_estimators = args.numTrees, criterion = args.impurity, max_depth = args.maxDepth, n_jobs = 1, verbose = 1)
    print(f'training {args.numTrees} {args.maxDepth} --  binary classification: {args.doBinaryClassification} -- pca: {args.usingPCA}')
    ert.fit(X_train, y_train)
    elapsed_time['training'] += time.time() - reference_time

    # TRAINING SUBSET
    # Make predictions
    y_true = y_train
    reference_time = time.time()
    y_pred = ert.predict(X_train)
    elapsed_time['inference_train'] += time.time() - reference_time

    filename_prefix = 'ert_%s_%05d_maxdepth_%03d' % (args.patient, args.numTrees, args.maxDepth)
    if args.usingPCA:
        filename_prefix += '_pca'
    else:
        filename_prefix += '_no_pca'
    if args.doBinaryClassification:
        filename_prefix += '_02_classes'
    else:
        filename_prefix += '_10_classes'
    save_results(f'{results_dir}.train/ert/{args.patient}', filename_prefix = filename_prefix, y_true = y_true, y_pred = y_pred, elapsed_time = elapsed_time['inference_train'], labels = labels)

    # Compute the train error
    error = sum(y_true != y_pred) / len(y_true)
    print(f"Train Error = {error}")

    # TESTING SUBSET
    # Make predictions
    y_true = y_test
    reference_time = time.time()
    y_pred = ert.predict(X_test)
    elapsed_time['inference_test'] += time.time() - reference_time

    save_results(f'{results_dir}.test/ert/{args.patient}', filename_prefix = filename_prefix, y_true = y_true, y_pred = y_pred, elapsed_time = elapsed_time['inference_test'], labels = labels)

    # Compute the test error
    error = sum(y_true != y_pred) / len(y_true)
    print(f"Test Error = {error}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('patient', type=str, help='Patient identifier')
    ''' For future versions
    parser.add_argument('--doBinaryClassification', default=False, type=bool,
                            action=argparse.BooleanOptionalAction,
                            help='Flag to indicate whether do the binary classification, default is false')
    '''
    parser.add_argument('--doBinaryClassification', action='store_true')
    parser.add_argument('--no-doBinaryClassification', action='store_false')
    parser.set_defaults(doBinaryClassification = False)
    parser.add_argument('--usingPCA', action='store_true')
    parser.add_argument('--no-usingPCA', action='store_false')
    parser.set_defaults(usingPCA = False)
    parser.add_argument('--verbose', default=0, type=int, help='Verbosity level')
    parser.add_argument('--numTrees', default=100, type=int, help='Number of trees in the ERT ensemble')
    parser.add_argument('--maxDepth', default=7, type=int, help='Max depth of each tree in the ERT ensemble')
    parser.add_argument('--impurity', default="gini", type=str, help='Impurity type. Options are: gini or entropy')

    parser.add_argument('--baseDir',    default='.',               type=str, help='Directory base from which create the directories for models, results and logs')
    parser.add_argument('--modelsDir',  default='models.l2.uc13',  type=str, help='Directory to save models --if it is the case')
    parser.add_argument('--resultsDir', default='results.l2.uc13', type=str, help='Directory where to store the results')
    parser.add_argument('--logDir',     default='log.l2.uc13',     type=str, help='Directory where to store the logs --if it is the case')
    parser.add_argument('--dataDir',    default='/bigdata/disk/jon/deephealth/uc13/21x20', type=str, help='Directory from which to load data in the local filesystem')

    main(parser.parse_args())
