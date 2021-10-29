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
import numpy
import pickle

from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
from matplotlib import pyplot

from utils_for_results import save_results
from KNN_Classifier import KNN_Classifier

try:
    from pyspark import SparkContext
    from pyspark.mllib.clustering import KMeans, KMeansModel
except:
    SparkContext = None


if __name__ == "__main__":
    """
    Usage: spark-submit --master local[4]  python/knn_uc13.py --k <k>
    """

    home_dir = os.getenv('HOME')
    if home_dir is None:
        raise Exception("Impossible to continue without the user's home directory")

    f = os.popen('hostname')
    hostname = f.readline().strip()
    f.close()
    if hostname is not None and hostname.find('eibds01') >= 0:
        hdfs_home_dir = '/user/cluster'
        local_home_dir = '/home/cluster'
    else:
        hdfs_home_dir = '/user/ubuntu'
        local_home_dir = '/home/ubuntu'
    
    verbose = 0
    spark_context = None
    num_partitions = 80
    global_patient = 'chb01'
    do_classification = False
    do_prediction = False
    K = 7
    label_mapping = [i for i in range(10)]
    version = 3
                                                   
    for i in range(len(sys.argv)):
        #if   sys.argv[i] == "--dataset"       :  dataset_filename = sys.argv[i + 1]
        #elif sys.argv[i] == "--results-dir"   :       results_dir = sys.argv[i + 1]
        #elif sys.argv[i] == "--models-dir"    :        models_dir = sys.argv[i + 1]
        #elif sys.argv[i] == "--log-dir"       :           log_dir = sys.argv[i + 1]
        if sys.argv[i] in ['--dataset', '--results-dir', '--models-dir', '--log-dir']:
            print()
            print(f'Option {sys.argv[i]} no longer accepted, for this code use --patient')
            print()
            sys.exit(0)

        elif sys.argv[i] == "--patient"       :    global_patient = sys.argv[i + 1]
        elif sys.argv[i] == "--verbosity"     :           verbose = int(sys.argv[i + 1])
        elif sys.argv[i] == "--num-partitions":    num_partitions = int(sys.argv[i + 1])
        elif sys.argv[i] == "--k"             :                 K = int(sys.argv[i + 1])
        elif sys.argv[i] == "--classify"      : do_classification = True
        elif sys.argv[i] == "--predict"       :     do_prediction = True
        elif sys.argv[i] == "--reduce-labels" :
            label_mapping[0] = 0
            label_mapping[1] = 1
            label_mapping[2] = 1
            label_mapping[3] = 0
            label_mapping[4] = 0
            label_mapping[5] = 0
            label_mapping[6] = 1
            label_mapping[7] = 0
            label_mapping[8] = 0
            label_mapping[9] = 0
            version = 4

    results_dir = f'{home_dir}/uc13.1/{global_patient}/results.{version}'
    models_dir = f'uc13.1/{global_patient}/models.{version}'
    log_dir = f'{home_dir}/uc13.1/{global_patient}/log.{version}'
    train_dataset_filename = f'{hdfs_home_dir}/data/uc13-pca-train.csv'
    test_dataset_filename = f'{hdfs_home_dir}/data/uc13-pca-test.csv'

    if SparkContext is not None:
        spark_context = SparkContext(appName = "K-Nearest-Neighbours-dataset-UC13")

    os.makedirs(log_dir,    exist_ok = True)
    #os.makedirs(models_dir, exist_ok = True)

    '''
        Load all the lines from a file (or files in a directory) into an RDD of text lines.

        It is assumed there is no header, each CSV file contains an undefined number or lines.
        - Each line represents a sample.
        - All the lines **must** contain the same number of values.
        - All the values **must** be numeric, i.e., integers or real values.
    '''
    # Data loading and parsing
    def csv_line_to_tuple(line):
        parts = line.split(';')
        patient = parts[0]
        label = label_mapping[int(parts[1])]
        sample = numpy.array([float(x) for x in parts[2:]])
        return (patient, label, sample)

    data = list()
    if global_patient < 'chb15':
        f = os.popen(f'hdfs dfs -cat {train_dataset_filename}')
    else:
        f = os.popen(f'hdfs dfs -cat {test_dataset_filename}')
    for line in f:
        t = csv_line_to_tuple(line)
        if t[0] == global_patient:
            data.append((t[1], t[2])) # patient is removed
    f.close()
    print(f"data has been loaded: {len(data)} samples")

    cut_point = int(0.7 * len(data))
    num_train_samples = cut_point
    num_test_samples = len(data) - num_train_samples
    dim = data[0][1].shape[0]

    # RDD creation and repartitioning
    if spark_context is not None:
        rdd_train_data = spark_context.parallelize(data[:cut_point], num_partitions)
        rdd_test_data  = spark_context.parallelize(data[cut_point:], num_partitions)
        print(f'loaded {num_train_samples} {dim}-dimensional samples for training into {rdd_train_data.getNumPartitions()} partitions')
        print(f'loaded {num_test_samples} {dim}-dimensional samples for testing into {rdd_test_data.getNumPartitions()} partitions')
    else:
        print(f'loaded {num_train_samples} {dim}-dimensional samples for training')
        print(f'loaded {num_test_samples} {dim}-dimensional samples for testing')
 
    #y_train = numpy.array(rdd_train_data.map(lambda t: t[0]).collect())
    #X_train = numpy.array(rdd_train_data.map(lambda t: t[1]).collect())
    y_train = numpy.array([t[0] for t in data[:cut_point]])
    X_train = numpy.array([t[1] for t in data[:cut_point]])
    y_test  = numpy.array([t[0] for t in data[cut_point:]])
    X_test  = numpy.array([t[1] for t in data[cut_point:]])

    elapsed_time = {'train' : 0, 'test' : 0}

    knn = KNN_Classifier(K = K, num_classes = len(numpy.unique(label_mapping)))
    knn.fit(X_train, y_train, min_samples_to_split = None)

    if spark_context is not None:
        for subset, rdd_data in zip(['train', 'test'], [rdd_train_data, rdd_test_data]):
            print(subset, rdd_data.count(), rdd_data.getNumPartitions())
            reference_time = time.time()
            y_true_pred = knn.predict(rdd_data)
            elapsed_time[subset] += time.time() - reference_time
            y_true = numpy.array([t[0] for t in y_true_pred])
            y_pred = numpy.array([t[1] for t in y_true_pred])
            #
            print(confusion_matrix(y_true, y_pred))
            print(classification_report(y_true, y_pred))
            # Save results in text and graphically represented confusion matrices
            filename_prefix = f'knn-classification-results-k-%d' % K
            save_results(f'{results_dir}-{subset}', filename_prefix, y_true, y_pred, elapsed_time = elapsed_time[subset])
    else:
        for subset, X, y_true in zip(['train', 'test'], [X_train, X_test], [y_train, y_test]):
            print(subset, X.shape, y_true.shape)
            reference_time = time.time()
            y_pred = knn.predict(X)
            elapsed_time[subset] += time.time() - reference_time
            print(confusion_matrix(y_true, y_pred))
            print(classification_report(y_true, y_pred))
            # Save results in text and graphically represented confusion matrices
            filename_prefix = f'knn-classification-results-k-%d' % K
            save_results(f'{results_dir}-{subset}', filename_prefix, y_true, y_pred, elapsed_time = elapsed_time[subset])

    if spark_context is not None:
        spark_context.stop()
