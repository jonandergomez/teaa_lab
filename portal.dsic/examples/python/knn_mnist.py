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

try:
    from pyspark import SparkContext, SparkConf
except:
    SparkContext = None
    SparkConf = None

from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from load_mnist import load_mnist
from utils_for_results import save_results
from machine_learning import KMeans

from KNN_Classifier import KNN_Classifier


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
    models_dir = f'{home_dir}/digits/models.3'
    log_dir = f'{home_dir}/digits/log.3'
    results_dir = f'{home_dir}/digits/results.3'
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


    if SparkConf is not None:
        spark_conf = SparkConf().set("spark.driver.maxResultSize", "24g").set("spark.app.name", "K-NearestNeighbors-with-dataset-MNIST")
        spark_context = SparkContext(conf = spark_conf)

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

    if model_filename is None:
        model_filename = f'knn-pca-{pca_components}'
        if use_kmeans:
            model_filename += f'-codebook-size-{codebook_size}'
        else:
            model_filename += '-no-kmeans'

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
        if spark_context is not None:
            rdd_train = spark_context.parallelize(codebooks, numSlices = num_partitions)
        else:
            y = list()
            X = list()
            for t in codebooks:
                y.append(t[0])
                X.append(t[1])
            rdd_train = (numpy.array(y), numpy.array(X))
        #############################################################################################################################
    else:
        if spark_context is not None:
            rdd_train = spark_context.parallelize([(y, x.copy()) for x, y in zip(X_train, y_train)], numSlices = num_partitions)
        else:
            rdd_train = (y_train, X_train)
    if spark_context is not None:
        rdd_test = spark_context.parallelize([(y, x.copy()) for x, y in zip(X_test,  y_test)],  numSlices = num_partitions)
        num_samples = rdd_train.count()
        print(f'train subset with {num_samples} distributed into {rdd_train.getNumPartitions()} partitions')
        print(f'test  subset with {rdd_test.count()} distributed into {rdd_test.getNumPartitions()} partitions')
    else:
        rdd_test = (y_test, X_test)
        print(f'train subset with {len(rdd_train[0])}')
        print(f'test  subset with {len(rdd_test[0])}')

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
            while len(c) < K and len(a) * len(b) > 0:
                if a[0][1] <= b[0][1]:
                    c.append(a[0])
                    del a[0]
                else: 
                    c.append(b[0])
                    del b[0]
            while len(c) < K and len(a) > 0:
                c.append(a[0])
                del a[0]
            while len(c) < K and len(b) > 0:
                c.append(b[0])
                del b[0]
            #del a, b
            '''
            Slower version using sort method from lists
            c = a + b
            c.sort(key = lambda x: x[1])
            if len(c) > K: c = c[:K]
            '''
            if len(c) > K:
                raise Exception('Unexpected length of a merged list')
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
        
    use_model = True
    if use_model:
        if os.path.exists(f'{models_dir}/{model_filename}'):
            with open(f'{models_dir}/{model_filename}', 'rb') as f:
                knn = pickle.load(f)
                f.close()
        else:
            os.makedirs(models_dir, exist_ok = True)
            knn = KNN_Classifier(K = K, num_classes = 10)
            knn.fit(X_train, y_train, min_samples_to_split = 100)
            with open(f'{models_dir}/{model_filename}', 'wb') as f:
                pickle.dump(knn, f)
                f.close()
        #
        if spark_context is not None:
            y_train_true_pred = knn.predict(rdd_train)
            y_test_true_pred = knn.predict(rdd_test)
            #
            y_train_pred = numpy.array([t[1] for t in y_train_true_pred])
            y_test_pred  = numpy.array([t[1] for t in y_test_true_pred])
        else:
            y_train_pred = knn.predict(X_train)
            y_test_pred  = knn.predict(X_test)
    else:
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
    if use_kmeans:
        filename_prefix += f'-codebook-size-{codebook_size}'
    else:
        filename_prefix += '-no-kmeans'

    save_results(f'{results_dir}.train', filename_prefix, y_train, y_train_pred)
    save_results(f'{results_dir}.test',  filename_prefix, y_test,  y_test_pred)
    #
    if spark_context is not None:
        spark_context.stop()
