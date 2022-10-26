"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://personales.upv.es/jon)
    Version: 1.0
    Date: October 2022
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

    Using ensembles based on decision trees for classification or regression

    Ensemble types:
        Random Forest
        Gradient Boosted Trees
        Extremely Randomized Trees

    This code is only for Extremely Randomized Trees (or Extra Trees), see the files
        rf_mnist.py  for Random Forest 
        gbt_mnist.py for Gradient Boosted Trees
"""

import sys
import os
import time
import argparse
import numpy

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.decomposition import PCA

from load_mnist import load_mnist
from utils_for_results import save_results


def main(args):

    X, y = load_mnist()
    X /= 255.0
    print(X.shape, y.shape)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    #
    pca = PCA(n_components = args.pcaComponents if args.pcaComponents <= 1.0 else int(args.pcaComponents))
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    pcaComponents = X_train.shape[1]

    labels = numpy.unique(y_train)


    results_dir = f'{args.baseDir}/{args.resultsDir}'
    models_dir  = f'{args.baseDir}/{args.modelsDir}'
    log_dir     = f'{args.baseDir}/{args.logDir}'
    #os.makedirs(log_dir,     exist_ok = True)
    #os.makedirs(models_dir,  exist_ok = True)
    #os.makedirs(results_dir, exist_ok = True)

    for nt in args.numTrees.split(sep = ':'):
        if nt is None or len(nt) == 0: continue
        numTrees = int(nt)

        for md in args.maxDepth.split(sep = ':'):
            if md is None or len(md) == 0: continue
            maxDepth = int(md)

            print('numTrees', numTrees, 'maxDepth', maxDepth, 'impurity', args.impurity)

            # Creating the ExtraTrees model
            ert = ExtraTreesClassifier(n_estimators = numTrees, criterion = args.impurity, max_depth = maxDepth, n_jobs = 1, verbose = 1)

            ert.fit(X_train, y_train)

            elapsed_time = {'train': 0, 'test': 0}

            # TRAINING SUBSET
            # Make predictions
            y_true = y_train
            reference_time = time.time()
            y_pred = ert.predict(X_train)
            elapsed_time['train'] += time.time() - reference_time

            filename_prefix = 'ert_%05d_pca_%04d_maxdepth_%03d' % (numTrees, pcaComponents, maxDepth)
            save_results(f'{results_dir}.train/ert', filename_prefix = filename_prefix, y_true = y_true, y_pred = y_pred, elapsed_time = None, labels = labels)


            # TESTING SUBSET
            # Make predictions.
            y_true = y_test
            reference_time = time.time()
            y_pred = ert.predict(X_test)
            elapsed_time['test'] += time.time() - reference_time

            filename_prefix = 'ert_%05d_pca_%04d_maxdepth_%03d' % (numTrees, pcaComponents, maxDepth)
            save_results(f'{results_dir}.test/ert', filename_prefix = filename_prefix, y_true = y_true, y_pred = y_pred, elapsed_time = None, labels = labels)
        # end for max depth
    # end for num trees



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', default=0, type=int, help='Verbosity level')
    parser.add_argument('--numTrees', default="100:200", type=str, help='Colon separated list of number of trees in the Random Forest')
    parser.add_argument('--maxDepth', default="5:7", type=str, help='Colon separated list of the max depth of each tree in the Random Forest')
    parser.add_argument('--impurity', default="gini", type=str, help='Impurity type. Options are: gini or entropy')
    parser.add_argument('--pcaComponents', default=37, type=float, help='Number of components of PCA an integer > 1 or a float in the range [0,1[')
    parser.add_argument('--baseDir',    default='.',                type=str, help='Directory base from which create the directories for models, results and logs')
    parser.add_argument('--modelsDir',  default='models.l2.mnist',  type=str, help='Directory to save models --if it is the case')
    parser.add_argument('--resultsDir', default='results.l2.mnist', type=str, help='Directory where to store the results')
    parser.add_argument('--logDir',     default='log.l2.mnist',     type=str, help='Directory where to store the logs --if it is the case')

    main(parser.parse_args())
