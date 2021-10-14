"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://personales.upv.es/jon)
    Version: 1.0
    Date: October 2021
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

    Using K-Nearest Neighbours for classification 
"""

import os
import sys
import time
import math
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
from machine_learning import KMeans


if __name__ == "__main__":
    """
    Usage: spark-submit --master local[4]  python/knn_mnist.py --k <k> 
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
    K = 7
    use_kmeans = False
    codebook_size = 200
    use_probs = False
                                                   
    for i in range(len(sys.argv)):
        if   sys.argv[i] == "--verbosity"     :           verbose = int(sys.argv[i + 1])
        elif sys.argv[i] == "--num-partitions":    num_partitions = int(sys.argv[i + 1])
        elif sys.argv[i] == "--k"             :                 K = int(sys.argv[i + 1])
        elif sys.argv[i] == "--codebook-size" :     codebook_size = int(sys.argv[i + 1])
        elif sys.argv[i] == "--use-kmeans"    :        use_kmeans = True
        elif sys.argv[i] == "--use-probs"     :         use_probs = True
        elif sys.argv[i] == "--model"         :    model_filename = sys.argv[i + 1]
        elif sys.argv[i] == "--train"         :       do_training = True
        elif sys.argv[i] == "--classify"      : do_classification = True
        elif sys.argv[i] == "--results-dir"   :       results_dir = sys.argv[i + 1]
        elif sys.argv[i] == "--models-dir"    :        models_dir = sys.argv[i + 1]
        elif sys.argv[i] == "--log-dir"       :           log_dir = sys.argv[i + 1]
        elif sys.argv[i] == "--pca"           :    pca_components = float(sys.argv[i + 1])
        elif sys.argv[i] == "--pf-degree"     :         pf_degree = int(sys.argv[i + 1])

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

    if use_kmeans:
        #############################################################################################################################
        codebooks = list()
        starting_time = time.time()
        for k in range(10):
            kmodel = KMeans(n_clusters = 200, verbosity = 1, modality = 'Lloyd', init = 'KMeans++')
            kmodel.epsilon = 1.0e-8
            kmodel.fit(X_train[y_train == k])
            for i in range(kmodel.n_clusters):
                codebooks.append((k, kmodel.cluster_centers_[i].copy()))
        print('processing time lapse for', kmodel.n_clusters, 'clusters per class', time.time() - starting_time, 'seconds')
        rdd_train = spark_context.parallelize(codebooks, numSlices = num_partitions)
        #############################################################################################################################
    else:
        rdd_train = spark_context.parallelize([(y, x.copy()) for x, y in zip(X_train, y_train)], numSlices = num_partitions)
    rdd_test  = spark_context.parallelize([(y, x.copy()) for x, y in zip(X_test,  y_test)],  numSlices = num_partitions)

    num_samples = rdd_train.count()

    print(f'train subset with {num_samples} distributed into {rdd_train.getNumPartitions()} partitions')
    print(f'test  subset with {rdd_test.count()} distributed into {rdd_test.getNumPartitions()} partitions')

    labels = numpy.unique(y_train)

    def compute_distances_for_numpy(x, sample):
        # x[0] contains the label
        # x[1] contains the sample
        result = list()
        for z in sample:
            result.append([(x[0], ((z - x[1]) ** 2).sum())])
        return result

    def compute_distances_for_lists(x, sample):
        # x[0] contains the label
        # x[1] contains the sample
        result = list()
        for z in sample:
            result.append([(x[0], ((z - x[1]) ** 2).sum())])
        return result

    def compute_distances_for_one_sample(x, sample):
        # x[0] contains the label
        # x[1] contains the sample
        return [[(x[0], ((sample - x[1]) ** 2).sum())]]

    def natural_merge(la, lb):
        lc = list()
        for i in range(len(la)):
            a = la[i]
            b = lb[i]
            c = list()
            #if type(a) is not list: raise Exception(type(a))
            #if type(b) is not list: raise Exception(type(b))
            while len(c) < K and len(a) > 0 and len(b) > 0:
                if a[0][1] <= b[0][1]:
                    c.append(a[0])
                    del a[0]
                else: 
                    c.append(b[0])
                    del b[0]
            #del a, b
            '''
            Slower version using sort method from lists
            c = a + b
            c.sort(key = lambda x: x[1])
            if len(c) > K: c = c[:K]
            '''
            lc.append(c)
        return lc

    def get_prediction(knn):
        y = [0] * len(labels)
        for k in knn: y[k[0]] += 1
        return numpy.array(y).argmax()

    def get_probs(knn):
        y = [0] * len(labels)
        for k in knn: y[k[0]] += 1
        y = numpy.array(y)
        return y / (1.0e-5 + y.sum())
        
    # Classifies the samples from the training subset
    knn = rdd_train.map(lambda x: compute_distances_for_numpy(x, X_train)).reduce(natural_merge)
    if use_probs:
        y_train_pred = numpy.array([get_probs(y) for y in knn]).argmax(axis = 1)
    else:
        y_train_pred = numpy.array([get_prediction(y) for y in knn])

    # Classifies the samples from the testing subset
    knn = rdd_train.map(lambda x: compute_distances_for_numpy(x, X_test)).reduce(natural_merge)
    if use_probs:
        y_test_pred = numpy.array([get_probs(y) for y in knn]).argmax(axis = 1)
    else:
        y_test_pred = numpy.array([get_prediction(y) for y in knn])

    # Save results in text and graphically represented confusion matrices
    filename_prefix = f'knn-classification-results-pca-{pca_components}-k-%d' % K
    save_results(f'{results_dir}.train', filename_prefix, y_train, y_train_pred)
    save_results(f'{results_dir}.test',  filename_prefix, y_test,  y_test_pred)
    #
    spark_context.stop()
