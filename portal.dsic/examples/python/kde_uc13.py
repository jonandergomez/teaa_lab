"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://personales.upv.es/jon)
    Version: 1.0
    Date: October 2021
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

    Using ensembles based on decision trees for classification or regression
    Ensemble types:
        Random Forest
        Gradient Boosted Trees
        Extremely Randomized Trees
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
from KernelClassifier import KernelClassifier

from pyspark import SparkContext


if __name__ == "__main__":
    """
    Usage: spark-submit --master local[4]  python/kde_uc13.py --band-width <bw>
    """

    home_dir = os.getenv('HOME')
    if home_dir is None:
        raise Exception("Impossible to continue without the user's home directory")

    f = os.popen('hostname')
    hostname = f.readline().strip()
    f.close()
    if hostname is not None and hostname.find('eibds01') >= 0:
        hdfs_home_dir = '/user/cluster'
    else:
        hdfs_home_dir = '/user/ubuntu'
    
    verbose = 0
    spark_context = None
    num_partitions = 80
    num_channels = 21
    do_classification = False
    do_prediction = False
    band_width = None
    label_mapping = [i for i in range(10)]
                                                   
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

        elif sys.argv[i] == "--verbosity"     :           verbose = int(sys.argv[i + 1])
        elif sys.argv[i] == "--num-partitions":    num_partitions = int(sys.argv[i + 1])
        elif sys.argv[i] == "--band-width"    :        band_width = float(sys.argv[i + 1])
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

    results_dir = f'{home_dir}/uc13.1/results.3'
    models_dir = f'uc13.1/models.3'
    log_dir = f'{home_dir}/uc13.1/log.3'
    train_dataset_filename = f'{hdfs_home_dir}/data/uc13-pca-train.csv'
    test_dataset_filename = f'{hdfs_home_dir}/data/uc13-pca-test.csv'


    spark_context = SparkContext(appName = "Kernel-Density-Estimation-dataset-UC13")

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
    train_csv_lines = spark_context.textFile(train_dataset_filename)
    test_csv_lines  = spark_context.textFile(test_dataset_filename)
    print("file(s) loaded ")
    # RDD repartitioning
    train_csv_lines = train_csv_lines.repartition(num_partitions)
    test_csv_lines  =  test_csv_lines.repartition(num_partitions)

    # RDD object conversion
    def csv_line_to_tuple(line):
        parts = line.split(';')
        label = label_mapping[int(parts[1])]
        sample = numpy.array([float(x) for x in parts[2:]])
        return (label, sample)

    rdd_train_data    = train_csv_lines.map(csv_line_to_tuple)
    rdd_test_data     = test_csv_lines.map(csv_line_to_tuple)

    # Retrives basic info from the RDD object
    num_train_samples = rdd_train_data.count()
    x = rdd_train_data.take(1)
    dim = x[0][1].shape[0]
    num_test_samples = rdd_test_data.count()

    print(f'loaded {num_train_samples} {dim}-dimensional samples for training into {rdd_train_data.getNumPartitions()} partitions')
    print(f'loaded {num_test_samples} {dim}-dimensional samples for testing  into {rdd_test_data.getNumPartitions()} partitions')
 

    # -------------------------------------------------------------------------------------------------------
    model = KernelClassifier(spark_context, band_width = band_width)
    model.fit(rdd_train_data)
    band_width = model.band_width
    # -------------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------------
    def predict_block_of_samples(temp_samples, y_train_pred, verbose = 0):
        pred_probs = model.predict_probs(temp_samples)
        if verbose > 0: print(pred_probs.shape, time.time())
        for i in range(0, len(pred_probs), num_channels):
            y_train_pred.append(pred_probs[i * num_channels : (i + 1) * num_channels].sum(axis = 0).argmax())
    # -------------------------------------------------------------------------------------------------------

    for subset, rdd_data in zip(['train', 'test'], [rdd_train_data, rdd_test_data]):
        print(subset, rdd_data.count(), rdd_data.getNumPartitions())
        y_true_pred = model.predict(rdd_data).collect()
        y_true = numpy.array([t[0] for t in y_true_pred])
        y_pred = numpy.array([t[1] for t in y_true_pred])
        #
        print(confusion_matrix(y_true, y_pred))
        print(classification_report(y_true, y_pred))
        # Save results in text and graphically represented confusion matrices
        filename_prefix = f'kde-classification-results-bw-%.3f' % band_width
        save_results(f'{results_dir}-{subset}', filename_prefix, y_true, y_pred)

    model.unpersist()
    spark_context.stop()

