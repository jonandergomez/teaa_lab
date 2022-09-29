"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://personales.upv.es/jon)
    Version: 1.0
    Date: October 2022
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

    Using E-M algorithm for Unsupervised Maximum Likelihood Estimation

"""

import os
import sys
import numpy
import random
import argparse

import machine_learning
from utils_for_results import save_results
from load_mnist import load_mnist
from sklearn.decomposition import PCA

from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import GaussianMixture


if __name__ == "__main__":
    """
    Usage: spark-submit --master local[4]  python/gmm_mnist_2022.py  --base-dir .   --dataset data/uc13-train.csv  --covar full      2>/dev/null
           spark-submit --master local[4]  python/gmm_mnist_2022.py  --base-dir .   --dataset data/uc13-train.csv  --covar diagonal  2>/dev/null

    Parameters
    ----------
    :param k:                Number of mixture components
    :param convergenceTol:   Convergence threshold. Default to 1e-3
    :param maxIterations:    Number of EM iterations to perform. Default to 200
    :param seed:             Random seed
    """

    #:param inputFile:        Input file path which contains data points
    parser = argparse.ArgumentParser()
    #parser.add_argument('inputFile', help='Input File')
    parser.add_argument('k', type=int, help='Number of clusters per digit')
    parser.add_argument('--convergenceTol', default=1e-3, type=float, help='convergence threshold')
    parser.add_argument('--maxIterations', default=200, type=int, help='Number of iterations')
    parser.add_argument('--seed', default=random.getrandbits(19), type=int, help='Random seed')
    parser.add_argument('--verbose', default=0, type=int, help='Verbosity level')
    parser.add_argument('--covarType', default='diagonal', type=str, help='Covariance type: diagonal or full')
    parser.add_argument('--minVar', default=1.0e-1, type=float, help='Min variance to avoid singular or non-positive definite covariance matrices')
    parser.add_argument('--pcaComponents', default=37, type=float, help='Number of components of PCA an integer > 1 or a float in the range [0,1[')
    parser.add_argument('--computeConfusionMatrix', default=False, type=bool, help='Flag to indicate whether compute the confusion matrix')
    parser.add_argument('--doClassification', default=False, type=bool, help='Flag to indicate whether do the classification')
    #
    parser.add_argument('--baseDir', default='.', type=str, help='Directory from which all the other paths are relative to')
    parser.add_argument('--resultsDir', default='results.l1.digits.2022', type=str, help='Directory where to store the results')
    parser.add_argument('--modelsDir', default='models.l1.digits.2022', type=str, help='Directory where to store the models')
    parser.add_argument('--logDir', default='log.l1.digits.2022', type=str, help='Directory where to store the logs')
    #
    parser.add_argument('--numPartitions', default=60, type=int, help='Number of partitions for RDDs')
    #
    args = parser.parse_args()

    gmm_filename = None

    spark_conf = SparkConf().set("spark.driver.maxResultSize", "24g").set("spark.app.name", "GMM-MLE-dataset-MNIST")
    spark_context = SparkContext(conf = spark_conf)

    X, y = load_mnist()
    X /= 255.0
    print(X.shape, y.shape)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    #
    pca = PCA(n_components = args.pcaComponents if args.pcaComponents<= 1.0 else int(args.pcaComponents))
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    os.makedirs(args.baseDir + '/' + args.logDir,     exist_ok = True)
    os.makedirs(args.baseDir + '/' + args.modelsDir,  exist_ok = True)
    os.makedirs(args.baseDir + '/' + args.resultsDir, exist_ok = True)

    results_dir = args.baseDir + '/' + args.resultsDir

    '''
    rdd_train = spark_context.parallelize([(y, x.copy()) for x, y in zip(X_train, y_train)], numSlices = args.numPartitions)
    num_samples = rdd_train.count()
    print(f'train subset with {num_samples} distributed into {rdd_train.getNumPartitions()} partitions')
    '''
    rdd_test  = spark_context.parallelize([(y, x.copy()) for x, y in zip(X_test,  y_test)],  numSlices = args.numPartitions)
    print(f'test  subset with {rdd_test.count()} distributed into {rdd_test.getNumPartitions()} partitions')


    labels = numpy.unique(y)
    print(labels)
    models = dict()

    for label in labels:
        data = spark_context.parallelize(X_train[y_train == label])
        print(f'training subset of digit {label} with {data.count()} distributed into {data.getNumPartitions()} partitions')
        gmm = machine_learning.GMM(n_components = args.k, dim = X_train.shape[1], covar_type = args.covarType, min_var = args.minVar)
        gmm.initialize_from(X_train[y_train == label])
        mle = machine_learning.MLE( covar_type = args.covarType,
                                    dim = gmm.dim,
                                    max_iterations = args.maxIterations,
                                    log_dir = args.baseDir + '/' + args.logDir,
                                    models_dir = args.baseDir + '/' + args.modelsDir)
        mle.gmm = gmm
        mle.fit_with_spark_one_gmm(spark_context = spark_context, samples = data, epsilon = args.convergenceTol)
        models[label] = mle.gmm

    def classify_sample(t):
        label, x = t
        p = list()
        for i in labels:
            gmm = models[i]
            _log_densities = gmm.log_densities(x, with_a_priori_probs = True)
            _densities = numpy.exp(_log_densities - _log_densities.max()) * numpy.exp(_log_densities.max())
            p.append(sum(_densities))
        p = numpy.array(p)
        return (label, p.argmax())


    y_true_and_pred = rdd_test.map(classify_sample).collect()
    y_true = numpy.array([x[0] for x in y_true_and_pred])
    y_pred = numpy.array([x[1] for x in y_true_and_pred])
    print('accuracy = ', 100 * sum(y_pred == y_true)  / len(y_true))

    filename_prefix = 'gmm-classification-results-k-%04d-pca-%s-min_var-%.6f-covar-%s' % (args.k, args.pcaComponents, args.minVar, args.covarType)
    save_results(f'{results_dir}.test', filename_prefix, y_true, y_pred)

    spark_context.stop()
