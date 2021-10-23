"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://personales.upv.es/jon)
    Version: 1.0
    Date: October 2021
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

    Using Kernel Density Estimation for classification 
"""

import os
import sys
import numpy
import pickle

from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
from matplotlib import pyplot

#from pyspark.mllib.stat import KernelDensity # generate errors when working with arrays instead of real values 

try:
    from pyspark import SparkContext
except:
    SparkContext = None

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
    models_dir = f'{home_dir}/digits/models.3'
    log_dir = f'{home_dir}/digits/log.3'
    results_dir = f'{home_dir}/digits/results.3'
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
    #    model_filename = f'{models_dir}/kde-bw-{band_width}'

    if SparkContext is not None:
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

    print(f'train subset with {len(X_train)} samples')
    print(f'test  subset with {len(X_test)} samples')

    model = KernelClassifier(band_width = band_width)
    model.fit(X_train, y_train)
    band_width = model.band_width

    if spark_context is not None:
        rdd_train = spark_context.parallelize([(y, x.copy()) for x, y in zip(X_train, y_train)], numSlices = num_partitions)
        rdd_test  = spark_context.parallelize([(y, x.copy()) for x, y in zip(X_test,  y_test)],  numSlices = num_partitions)
        num_samples = rdd_train.count()
        y_train_true, y_train_pred = model.predict(rdd_train)
        y_test_true,  y_test_pred  = model.predict(rdd_test)
    else:
        # Classifies the samples from the training subset
        y_train_true = y_train
        y_train_pred = model.predict(X_train)
        # Classifies the samples from the testing subset
        y_test_true = y_test
        y_test_pred = model.predict(X_test)

    # Save results in text and graphically represented confusion matrices
    filename_prefix = f'kde-classification-results-pca-{pca_components}-bw-%.3f' % band_width
    save_results(f'{results_dir}.train', filename_prefix, y_train_true, y_train_pred)
    save_results(f'{results_dir}.test',  filename_prefix, y_test_true,  y_test_pred)
    #
    if spark_context is not None:
        spark_context.stop()
