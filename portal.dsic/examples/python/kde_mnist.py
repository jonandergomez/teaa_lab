"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://personales.upv.es/jon)
    Version: 1.0
    Date: October 2021
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

    Using kernel density estimation for classification 
"""

import os
import sys
import numpy
import pickle

from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
from matplotlib import pyplot

#from pyspark.mllib.stat import KernelDensity # generate errors when working with arrays instead of real values 

from pyspark import SparkContext

from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from load_mnist import load_mnist
from utils_for_results import save_results
from KernelClassifier import KernelClassifier

if __name__ == "__main__":
    """
    Usage: spark-submit --master local[4]  python/kde_mnist.py --band-width <bw> 
    """

    home_dir = os.getenv('HOME')
    if home_dir is None:
        raise Exception("Impossible to continue without a reference to the user's home directory")

    verbose = 0
    spark_context = None
    num_partitions = 80
    models_dir = f'{home_dir}/models.digits.3'
    log_dir = f'{home_dir}/log.digits.3'
    results_dir = f'{home_dir}/results.digits.3'
    do_training = False
    do_classification = False
    model_filename = None
    band_width = None
    pca_components = 30
    pf_degree = 1
                                                   
    for i in range(len(sys.argv)):
        if   sys.argv[i] == "--verbosity"     :           verbose = int(sys.argv[i + 1])
        elif sys.argv[i] == "--num-partitions":    num_partitions = int(sys.argv[i + 1])
        elif sys.argv[i] == "--band-width"    :        band_width = float(sys.argv[i + 1])
        elif sys.argv[i] == "--model"         :    model_filename = sys.argv[i + 1]
        elif sys.argv[i] == "--train"         :       do_training = True
        elif sys.argv[i] == "--classify"      : do_classification = True
        elif sys.argv[i] == "--results-dir"   :       results_dir = sys.argv[i + 1]
        elif sys.argv[i] == "--models-dir"    :        models_dir = sys.argv[i + 1]
        elif sys.argv[i] == "--log-dir"       :           log_dir = sys.argv[i + 1]
        elif sys.argv[i] == "--pca"           :    pca_components = float(sys.argv[i + 1])
        elif sys.argv[i] == "--pf-degree"     :         pf_degree = int(sys.argv[i + 1])


    #if model_filename is None:
    #    model_filename = f'{models_dir}/kde-bd-{band_width}'

    spark_context = SparkContext(appName = "KernelDensityEstimation-with-dataset-MNIST")

    X, y = load_mnist()
    X /= 255.0
    print(X.shape, y.shape)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    #
    #pca = PCA(n_components = 0.95)
    if pca_components > 1: pca_components = int(pca_components)
    pca = PCA(n_components = pca_components)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    pca_components = X_train.shape[1]
    if pf_degree > 1:
        pf = PolynomialFeatures(degree = pf_degree, interaction_only = True, include_bias = False)
        X_train = pf.fit_transform(X_train)
        X_test = pf.fit_transform(X_test)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    '''
    from sklearn.utils import shuffle
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_train, y_train)
    X_train = X_train[:100]
    y_train = y_train[:100]
    X_test = X_test[:100]
    y_test = y_test[:100]
    '''

    rdd_train = spark_context.parallelize([(y, x.copy()) for x, y in zip(X_train, y_train)], numSlices = num_partitions)
    rdd_test  = spark_context.parallelize([(y, x.copy()) for x, y in zip(X_test,  y_test)],  numSlices = num_partitions)

    num_samples = rdd_train.count()

    print(f'train subset with {num_samples} distributed into {rdd_train.getNumPartitions()} partitions')
    print(f'test  subset with {rdd_test.count()} distributed into {rdd_test.getNumPartitions()} partitions')

    labels = numpy.unique(y_train)

    model = KernelClassifier(spark_context, band_width = band_width)
    model.fit(rdd_train)
    band_width = model.band_width

    # Classifies the samples from the training subset
    y_true_pred  = model.predict(rdd_train).collect()
    y_train      = numpy.array([t[0] for t in y_true_pred])
    y_train_pred = numpy.array([t[1] for t in y_true_pred])
    #y_train_pred = model.predict(X_train)
    #y_train_pred = numpy.array(y_train_pred)
    # Classifies the samples from the testing subset
    y_true_pred = model.predict(rdd_test).collect()
    y_test      = numpy.array([t[0] for t in y_true_pred])
    y_test_pred = numpy.array([t[1] for t in y_true_pred])
    #y_test_pred  = model.predict(X_test)
    #y_test_pred = numpy.array(y_test_pred)

    model.unpersist() # because subsets of samples per target class are persisted RDD

    # Save results in text and graphically represented confusion matrices
    filename_prefix = f'kde-classification-results-pca-{pca_components}-bd-%.3f' % band_width
    save_results(f'{results_dir}.train', filename_prefix, y_train, y_train_pred)
    save_results(f'{results_dir}.test',  filename_prefix, y_test,  y_test_pred)
    #
    spark_context.stop()
