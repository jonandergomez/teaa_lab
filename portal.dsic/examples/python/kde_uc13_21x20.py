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

    This code is only for Kernel Density Estimation Classifiers, see the files
        knn_uc13_21x20.py for K-Nearest Neighbours
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

from KernelClassifier import KernelClassifier
from utils_for_results import save_results


def main(args, sc):

    # Loading and parsing the data file, converting it to a DataFrame.
    if args.usingPCA:
        trainData = sc.read.option('delimiter', ';').csv(f"data/uc13/uc13-{args.patient}-21x20-train-pca.csv")
        testData  = sc.read.option('delimiter', ';').csv(f"data/uc13/uc13-{args.patient}-21x20-test-pca.csv")
    else:
        trainData = sc.read.option('delimiter', ';').csv(f"data/uc13/uc13-{args.patient}-21x20-train.csv")
        testData  = sc.read.option('delimiter', ';').csv(f"data/uc13/uc13-{args.patient}-21x20-test.csv")

    # Preparing the labels according to the number of target classes
    #   2 for binary classification
    #  10 for multi-class classification
    label_mapping = [i for i in range(10)]
    if args.doBinaryClassification:
        label_mapping = [0 for i in range(10)]
        label_mapping[1] = 1
    labels = numpy.unique(label_mapping)
        
    # Function to process each CSV line
    def process_csv_line(row):
        # returns a list with patient, label and features
        #return row[0], int(label_mapping[int(row[1])]), numpy.array([float(x) for x in row[2:]])
        # returns a list with label and features
        return int(label_mapping[int(row[1])]), numpy.array([float(x) for x in row[2:]])

    # Processing each csv line in parallel and regenerate DataFrames with the appropriate column names
    trainData = trainData.rdd.map(process_csv_line)#.toDF(['patient', 'label', 'features'])
    testData  =  testData.rdd.map(process_csv_line)#.toDF(['patient', 'label', 'features'])

    trainData.persist()
    testData.persist()

    yX = trainData.collect()
    X_train = numpy.array([t[1] for t in yX])
    y_train = numpy.array([t[0] for t in yX])
    del yX
    yX = testData.collect()
    X_test = numpy.array([t[1] for t in yX])
    y_test = numpy.array([t[0] for t in yX])
    del yX

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
                training_samples = X_train[y_train == k]
                if len(training_samples) > codebookSize:
                    kmodel = KMeans(n_clusters = codebookSize, verbosity = 1, modality = 'Lloyd', init = 'KMeans++')
                    kmodel.epsilon = 1.0e-8
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

        print(_X_train_.shape, _y_train_.shape)

        for bw in args.bandWidth.split(sep = ':'):
            if bw is None or len(bw) == 0: continue
            bandWidth = float(bw)

            print(args.patient, 'KMeans codebook size', codebookSize, 'bandWidth', bandWidth)

            train_elapsed_time = 0
            test_elapsed_time = 0
            reference_timestamp = time.time()

            # Creating the Kernel Density Estimation classifier
            kde = KernelClassifier(band_width = bandWidth)

            # Training the model
            kde.fit(_X_train_, _y_train_)

            train_elapsed_time += time.time() - reference_timestamp

            if sc is not None:
                rdd_train = trainData # sc.parallelize([(y, x.copy()) for x, y in zip(X_train, y_train)])
                rdd_test  = testData  # sc.parallelize([(y, x.copy()) for x, y in zip(X_test,  y_test)])
                # num_samples = rdd_train.count()

                # Classifies the samples from the training subset
                reference_timestamp = time.time()
                y_train_true, y_train_pred = kde.predict(rdd_train)
                train_elapsed_time += time.time() - reference_timestamp
                # Classifies the samples from the testing subset
                reference_timestamp = time.time()
                y_test_true,  y_test_pred  = kde.predict(rdd_test)
                test_elapsed_time += time.time() - reference_timestamp
            else:
                # Classifies the samples from the training subset
                reference_timestamp = time.time()
                y_train_true = y_train
                y_train_pred = kde.predict(X_train)
                train_elapsed_time += time.time() - reference_timestamp
                # Classifies the samples from the testing subset
                reference_timestamp = time.time()
                y_test_true = y_test
                y_test_pred = kde.predict(X_test)
                test_elapsed_time += time.time() - reference_timestamp

            filename_prefix = 'kde_kmeans_%04d_pca_%04d_bandwidth_%.3f_%02d_classes' % (codebookSize, pcaComponents, bandWidth, len(labels))
            save_results(f'{results_dir}.train/kde/{args.patient}', filename_prefix = filename_prefix, y_true = y_train_true, y_pred = y_train_pred, elapsed_time = train_elapsed_time, labels = labels)
            save_results(f'{results_dir}.test/kde/{args.patient}',  filename_prefix = filename_prefix, y_true = y_test_true,  y_pred = y_test_pred,  elapsed_time = test_elapsed_time,  labels = labels)

        # end for bandwidth
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
    parser.add_argument('--usingPCA', action='store_true')
    parser.add_argument('--no-usingPCA', action='store_false')
    parser.set_defaults(usingPCA = False)
    #
    parser.add_argument('--verbose', default=0, type=int, help='Verbosity level')
    parser.add_argument('--codebookSize', default="0:100:200", type=str, help='Colon separated list of the codebook sizes to apply kmeans before KDE')
    parser.add_argument('--bandWidth', default="0.1:0.2:0.5:1.0:2.0", type=str, help='Colon separated list of the band width for the KDE classifier')
    parser.add_argument('--baseDir',    default='.',                type=str, help='Directory base from which create the directories for models, results and logs')
    parser.add_argument('--modelsDir',  default='models.l3.uc13',  type=str, help='Directory to save models --if it is the case')
    parser.add_argument('--resultsDir', default='results.l3.uc13', type=str, help='Directory where to store the results')
    parser.add_argument('--logDir',     default='log.l3.uc13',     type=str, help='Directory where to store the logs --if it is the case')

    sc = SparkSession.builder.appName(f"KernelDensityEstimationClassifierForUC13").getOrCreate()
    #sc = SparkContext(appName = "KernelDensityEstimationClassifierForUC13")
    main(parser.parse_args(), sc)
    if sc is not None: sc.stop()
