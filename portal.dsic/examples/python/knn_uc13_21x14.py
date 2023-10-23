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
        kde_uc13_21x20.py for Kernel Density Estimation
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
from utils_for_results import save_results

from eeg_load_data import load_csv_from_uc13

def main(args, sc):

    #hdfs_url = 'hdfs://teaa-master-ubuntu22:8020'
    #num_partitions = 60
    hdfs_url = 'hdfs://eibds01.mbda:8020'
    num_partitions = 80

    do_z_transform = (args.format == '21x14')

    log_dir     = f'{args.baseDir}/{args.logDir}/knn'
    models_dir  = f'{args.baseDir}/{args.modelsDir}/knn'
    results_dir = f'{args.baseDir}/{args.resultsDir}/knn/{args.patient}'

    task = 'binary-classification' if args.doBinaryClassification else 'multi-class-classification'

    os.makedirs(log_dir,     exist_ok = True)
    os.makedirs(models_dir,  exist_ok = True)
    os.makedirs(results_dir, exist_ok = True)

    if args.patient == 'ALL':
        train_filenames = [f'{hdfs_url}/data/uc13/21x14/uc13-chb{i:02d}-{args.format}-time-to-seizure.csv' for i in range(1,17)]
        test_filenames  = [f'{hdfs_url}/data/uc13/21x14/uc13-chb{i:02d}-{args.format}-time-to-seizure.csv' for i in range(17,25)]
    else:
        train_filenames = [f'{hdfs_url}/data/uc13/21x14/uc13-{args.patient}-{args.format}-time-to-seizure-train.csv']
        test_filenames  = [f'{hdfs_url}/data/uc13/21x14/uc13-{args.patient}-{args.format}-time-to-seizure-test.csv']

    # Loads and repartitions the data
    rdd_train = load_csv_from_uc13(sc, train_filenames, num_partitions = num_partitions, do_binary_classification = args.doBinaryClassification)
    rdd_test  = load_csv_from_uc13(sc,  test_filenames, num_partitions = num_partitions, do_binary_classification = args.doBinaryClassification)

    # BEGIN: Perform the standard scalation
    if do_z_transform:
        mean = rdd_train.map(lambda sample: sample[3]).reduce(lambda x, y: x + y) / rdd_train.count()
        variance = rdd_train.map(lambda sample: (sample[3] - mean)**2).reduce(lambda x, y: x + y) / rdd_train.count()
        sigma = numpy.maximum(1.0e-3, numpy.sqrt(variance))
        #
        rdd_train = rdd_train.map(lambda sample: (sample[0], sample[1], sample[2], (sample[3] - mean) / sigma))
        rdd_test  =  rdd_test.map(lambda sample: (sample[0], sample[1], sample[2], (sample[3] - mean) / sigma))
    # END: Perform the standard scalation

    rdd_train.persist()
    rdd_test.persist()

    # each tuple in the RDD has (patient, time-to-seizure, label, features)

    yX = rdd_train.collect()
    X_train = numpy.array([t[3] for t in yX])
    y_train = numpy.array([t[2] for t in yX])
    del yX
    yX = rdd_test.collect()
    X_test = numpy.array([t[3] for t in yX])
    y_test = numpy.array([t[2] for t in yX])
    del yX

    labels = numpy.unique(y_train)
    print(f'labels: {labels} for task {task}')

    for cb_size in args.codebookSize.split(sep = ':'):
        if cb_size is None or len(cb_size) == 0: continue
        codebookSize = int(cb_size)

        kmeans_time = time.time()
        if codebookSize > 0:
            codebooks = list()
            for k in range(10):
                starting_time = time.time()
                training_samples = X_train[y_train == k]
                if len(training_samples) > codebookSize:
                    kmodel = KMeans(n_clusters = min(codebookSize, len(training_samples) // 20), verbosity = 0, modality = 'Lloyd', init = 'KMeans++')
                    kmodel.epsilon = 1.0e-6
                    kmodel.fit(X_train[y_train == k])
                    for i in range(kmodel.n_clusters):
                        codebooks.append((k, kmodel.cluster_centers_[i].copy()))
                    print('processing time lapse for', kmodel.n_clusters, 'clusters per class', time.time() - starting_time, 'seconds')
                else:
                    for i in range(len(training_samples)):
                        codebooks.append((k, training_samples[i].copy()))
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

        kmeans_time = time.time() - kmeans_time

        print(_X_train_.shape, _y_train_.shape)

        for _k_ in args.K.split(sep = ':'):
            if _k_ is None or len(_k_) == 0: continue
            K = int(_k_)

            print(args.patient, 'KMeans codebook size', codebookSize, 'K', K)

            train_elapsed_time = kmeans_time
            test_elapsed_time = 0
            reference_timestamp = time.time()

            # Creating the K-Nearest Neighbour classifier
            knn = KNN_Classifier(K = K, num_classes = len(labels))

            # Training the model
            knn.fit(_X_train_, _y_train_, min_samples_to_split = 100 if codebookSize == 0 else 50)

            train_elapsed_time += time.time() - reference_timestamp

            if sc is not None:
                # Classifies the samples from the training subset
                reference_timestamp = time.time()
                y_train_true_pred = knn.predict(rdd_train.map(lambda t: (t[2], t[3])))
                train_elapsed_time += time.time() - reference_timestamp
                y_train_true = numpy.array([t[0] for t in y_train_true_pred])
                y_train_pred = numpy.array([t[1] for t in y_train_true_pred])
                # Classifies the samples from the testing subset
                reference_timestamp = time.time()
                y_test_true_pred = knn.predict(rdd_test.map(lambda t: (t[2], t[3])))
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

            filename_prefix = 'knn_kmeans_%04d_%s_K_%03d_%s' % (codebookSize, args.format, K, task)
            save_results(f'{results_dir}/train', filename_prefix = filename_prefix, y_true = y_train_true, y_pred = y_train_pred, elapsed_time = train_elapsed_time, labels = labels)
            save_results(f'{results_dir}/test',  filename_prefix = filename_prefix, y_true = y_test_true,  y_pred = y_test_pred,  elapsed_time = test_elapsed_time,  labels = labels)

        # end for K
    # end for KMeans codebook size
# end of the method main()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #
    parser.add_argument('patient', type=str, help='Patient identifier')
    #
    parser.add_argument('--doBinaryClassification', action='store_true')
    parser.add_argument('--no-doBinaryClassification', action='store_false')
    parser.set_defaults(doBinaryClassification = False)
    #
    parser.add_argument('--format',  default='21x14',  type=str, help='Data format (e.g., if PCA is applied use pca136 or pca141')
    #
    parser.add_argument('--verbose', default=0, type=int, help='Verbosity level')
    parser.add_argument('--codebookSize', default="0:100:200", type=str, help='Colon separated list of the codebook sizes to apply kmeans before KNN')
    parser.add_argument('--K', default="3:5:7:9:11:13", type=str, help='Colon separated list of K, i.e.,  the number of thet nearest training samples to the one to be classified')
    parser.add_argument('--baseDir',    default='.',                type=str, help='Directory base from which create the directories for models, results and logs')
    parser.add_argument('--modelsDir',  default='models/uc13',  type=str, help='Directory to save models --if it is the case')
    parser.add_argument('--resultsDir', default='results/uc13', type=str, help='Directory where to store the results')
    parser.add_argument('--logDir',     default='logs/uc13',    type=str, help='Directory where to store the logs --if it is the case')

    args = parser.parse_args()
    #sc = SparkSession.builder.appName(f"knn-EEG-{args.format}").getOrCreate()
    sc = SparkContext(appName = f"knn-EEG-{args.format}")
    main(args, sc)
    if sc is not None: sc.stop()
