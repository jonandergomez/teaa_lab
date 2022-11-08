"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://personales.upv.es/jon)
    Version: 1.0
    Date: November 2022
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

    Using memory-based techniques for classification

    Memory-based techniques used in this lab practice:
        Kernel Density Estimation
        K-Nearest Neighbours

    This code is only for K-Nearest Neighbours Classifiers, see the file
        kde_mnist.py for Kernel Density Estimation
"""

import sys
import os
import time
import argparse
import numpy

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

from sklearn.decomposition import PCA
#from pyspark.mllib.clustering import KMeans, KMeansModel
from machine_learning import KMeans

from KNN_Classifier import KNN_Classifier
from load_mnist import load_mnist
from utils_for_results import save_results


def main(args, sc):

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

    for cb_size in args.codebookSize.split(sep = ':'):
        if cb_size is None or len(cb_size) == 0: continue
        codebookSize = int(cb_size)

        if codebookSize > 0:
            codebooks = list()
            for k in range(10):
                starting_time = time.time()
                kmodel = KMeans(n_clusters = codebookSize, verbosity = 1, modality = 'Lloyd', init = 'KMeans++')
                kmodel.epsilon = 1.0e-8
                kmodel.fit(X_train[y_train == k])
                for i in range(kmodel.n_clusters):
                    codebooks.append((k, kmodel.cluster_centers_[i].copy()))
                print('processing time lapse for', kmodel.n_clusters, 'clusters per class', time.time() - starting_time, 'seconds')
            y = list()
            X = list()
            for t in codebooks:
                y.append(t[0])
                X.append(t[1])
            _y_train_ = numpy.array(y)
            _X_train_ = numpy.array(X)
        else:
            _y_train_ = y_train
            _X_train_ = X_train

        print(_X_train_.shape, _y_train_.shape)

        for _k_ in args.K.split(sep = ':'):
            if _k_ is None or len(_k_) == 0: continue
            K = int(_k_)

            print('KMeans codebook size', codebookSize, 'K', K)

            train_elapsed_time = 0
            test_elapsed_time = 0
            reference_timestamp = time.time()

            # Creating the K-Nearest Neighbour classifier
            knn = KNN_Classifier(K = K, num_classes = len(labels))

            # Training the model
            knn.fit(_X_train_, _y_train_, min_samples_to_split = 100 if codebookSize == 0 else codebookSize // 20)

            train_elapsed_time += time.time() - reference_timestamp

            if sc is not None:
                rdd_train = sc.parallelize([(y, x.copy()) for x, y in zip(X_train, y_train)])
                rdd_test  = sc.parallelize([(y, x.copy()) for x, y in zip(X_test,  y_test)])
                # num_samples = rdd_train.count()

                # Classifies the samples from the training subset
                reference_timestamp = time.time()
                y_train_true_pred = knn.predict(rdd_train)
                train_elapsed_time += time.time() - reference_timestamp
                y_train_true = numpy.array([t[0] for t in y_train_true_pred])
                y_train_pred = numpy.array([t[1] for t in y_train_true_pred])
                # Classifies the samples from the testing subset
                reference_timestamp = time.time()
                y_test_true_pred = knn.predict(rdd_test)
                test_elapsed_time += time.time() - reference_timestamp
                y_test_true = numpy.array([t[0] for t in y_test_true_pred])
                y_test_pred = numpy.array([t[1] for t in y_test_true_pred])
            else:
                # Classifies the samples from the training subset
                reference_timestamp = time.time()
                y_train_true = y_train
                y_train_pred = knn.predict(X_train)
                train_elapsed_time += time.time() - reference_timestamp
                # Classifies the samples from the testing subset
                reference_timestamp = time.time()
                y_test_true = y_test
                y_test_pred = knn.predict(X_test)
                test_elapsed_time += time.time() - reference_timestamp

            filename_prefix = 'knn_kmeans_%04d_pca_%04d_K_%03d' % (codebookSize, pcaComponents, K)
            save_results(f'{results_dir}.train/knn', filename_prefix = filename_prefix, y_true = y_train_true, y_pred = y_train_pred, elapsed_time = train_elapsed_time, labels = labels)
            save_results(f'{results_dir}.test/knn',  filename_prefix = filename_prefix, y_true = y_test_true,  y_pred = y_test_pred,  elapsed_time = test_elapsed_time,  labels = labels)

        # end for K
    # end for KMeans codebook size
# end of the method main()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', default=0, type=int, help='Verbosity level')
    parser.add_argument('--codebookSize', default="0:100:200", type=str, help='Colon separated list of the codebook sizes to apply kmeans before KNN')
    parser.add_argument('--K', default="3:5:7:9:11:13", type=str, help='Colon separated list of K, i.e.,  the number of thet nearest training samples to the one to be classified')
    parser.add_argument('--pcaComponents', default=37, type=float, help='Number of components of PCA an integer > 1 or a float in the range [0,1[')
    parser.add_argument('--baseDir',    default='.',                type=str, help='Directory base from which create the directories for models, results and logs')
    parser.add_argument('--modelsDir',  default='models.l3.mnist',  type=str, help='Directory to save models --if it is the case')
    parser.add_argument('--resultsDir', default='results.l3.mnist', type=str, help='Directory where to store the results')
    parser.add_argument('--logDir',     default='log.l3.mnist',     type=str, help='Directory where to store the logs --if it is the case')

    #sc = SparkSession.builder.appName(f"KernelDensityEstimationClassifierForMNIST").getOrCreate()
    sc = SparkContext(appName = "KernelDensityEstimationClassifierForMNIST")
    main(parser.parse_args(), sc)
    if sc is not None: sc.stop()
