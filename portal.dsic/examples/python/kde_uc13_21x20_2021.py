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
import time
import numpy
import pickle

from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
from matplotlib import pyplot

from utils_for_results import save_results
from KernelClassifier import KernelClassifier

try:
    from pyspark import SparkContext
    #from pyspark.mllib.clustering import KMeans, KMeansModel
except:
    SparkContext = None


if __name__ == "__main__":
    """
    Usage: spark-submit --master local[4]  python/kde_uc13_21x20.py --band-width <bw>
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
    num_channels = 21
    global_patient = 'chb01'
    do_classification = False
    do_prediction = False
    band_width = None
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
            version = 4

    results_dir = f'{home_dir}/uc13-21x20/{global_patient}/results.{version}'
    models_dir = f'uc13-21x20/{global_patient}/models.{version}'
    log_dir = f'{home_dir}/uc13-21x20/{global_patient}/log.{version}'
    train_dataset_filename = f'{hdfs_home_dir}/data/uc13/uc13-{global_patient}-21x20-train.csv'
    test_dataset_filename = f'{hdfs_home_dir}/data/uc13/uc13-{global_patient}-21x20-test.csv'

    if SparkContext is not None:
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
    # RDD object conversion
    def csv_line_to_tuple(line):
        parts = line.split(';')
        label = label_mapping[int(parts[1])]
        sample = numpy.array([float(x) for x in parts[2:]]).reshape(num_channels, -1)
        return (label, sample)

    def csv_line_to_list_of_tuples(line):
        parts = line.split(';')
        label = label_mapping[int(parts[1])]
        sample = numpy.array([float(x) for x in parts[2:]]).reshape(num_channels, -1)
        return [(label, x) for x in sample]

    if spark_context is not None:
        # Data loading and parsing
        train_csv_lines = spark_context.textFile(train_dataset_filename)
        test_csv_lines  = spark_context.textFile(test_dataset_filename)
        print("file(s) loaded ")
        # RDD repartitioning
        train_csv_lines = train_csv_lines.repartition(num_partitions)
        test_csv_lines  =  test_csv_lines.repartition(num_partitions)

        rdd_train_data    = train_csv_lines.map(csv_line_to_tuple)#.sample(False, 0.01)
        rdd_train_samples = train_csv_lines.flatMap(csv_line_to_list_of_tuples)
        rdd_test_data     = test_csv_lines.map(csv_line_to_tuple)#.sample(False, 0.01)
        rdd_test_samples  = test_csv_lines.flatMap(csv_line_to_list_of_tuples)

        # Retrives basic info from the RDD object
        num_train_samples = rdd_train_samples.count()
        x = rdd_train_samples.take(1)
        dim = x[0][1].shape[0]
        num_test_samples = rdd_test_samples.count()

        print(f'loaded {num_train_samples} {dim}-dimensional samples for training into {rdd_train_samples.getNumPartitions()} partitions')
        print(f'loaded {num_test_samples} {dim}-dimensional samples for testing into {rdd_test_samples.getNumPartitions()} partitions')
 
        y_train = numpy.array(rdd_train_samples.map(lambda t: t[0]).collect())
        X_train = numpy.array(rdd_train_samples.map(lambda t: t[1]).collect())
    else:
        with os.popen(f'hdfs dfs -cat {train_dataset_filename}') as f:
            train_csv_lines = f.readlines()
            f.close()
        with os.popen(f'hdfs dfs -cat {test_dataset_filename}') as f:
            test_csv_lines = f.readlines()
            f.close()

        train_samples = list()
        for csv_line in train_csv_lines:
            train_samples += csv_line_to_list_of_tuples(csv_line)

        y_train = numpy.array([t[0] for t in train_samples])
        X_train = numpy.array([t[1] for t in train_samples])

    model = KernelClassifier(band_width = band_width)

    elapsed_time = {'train' : 0, 'test' : 0}

    use_kmeans = True
    n_clusters = 4000
    if use_kmeans:
        #############################################################################################################################
        from machine_learning import KMeans as JonKMeans, kmeans_load
        codebooks_y = list()
        codebooks_X = list()
        starting_time = time.time()
        for label in numpy.unique(y_train):
            print('computing k-means for label', label)
            _X_train = X_train[y_train == label]
            if len(_X_train) <= n_clusters:
                for i in range(len(_X_train)):
                    codebooks_y.append(label)
                    codebooks_X.append(_X_train[i].copy())
            else:
                os.makedirs(f'{local_home_dir}/{models_dir}', exist_ok = True)
                kmodel_filename = f'{local_home_dir}/{models_dir}/kmeans-%05d-%03d.pkl' % (n_clusters, label)
                kmodel = kmeans_load(kmodel_filename)
                if kmodel is None:
                    #kmodel = JonKMeans(n_clusters = n_clusters, verbosity = 1, modality = 'original-k-Means')
                    kmodel = JonKMeans(n_clusters = n_clusters, verbosity = 1, modality = 'Lloyd', init = 'random')
                    kmodel.epsilon = 1.0e-8
                    #kmodel.fit(_X_train)
                    #kmodel.modality = 'Lloyd'
                    kmodel.fit(_X_train)
                    #kmeans_model = KMeans.train(rdd_train_samples.map(lambda t: t[1]), k = n_clusters, maxIterations = 2000, initializationMode = "kmeans||", initializationSteps = 5, epsilon = 1.0e-9)
                    #kmodel.cluster_centers_ = numpy.array(kmeans_model.centers)
                    #kmodel.n_clusters = len(kmodel.cluster_centers_)
                    kmodel.save(kmodel_filename)
                #
                for i in range(kmodel.n_clusters):
                    codebooks_y.append(label)
                    codebooks_X.append(kmodel.cluster_centers_[i].copy())
        print('processing time lapse for', len(codebooks_X), 'clusters in total', time.time() - starting_time, 'seconds')
        codebooks_X = numpy.array(codebooks_X)
        codebooks_y = numpy.array(codebooks_y)
        model.fit(codebooks_X, codebooks_y)
        #############################################################################################################################
    else:            
        model.fit(X_train, y_train)
    #
    band_width = model.band_width

    if spark_context is not None:
        for subset, rdd_data in zip(['train', 'test'], [rdd_train_data, rdd_test_data]):
            print(subset, rdd_data.count(), rdd_data.getNumPartitions())
            reference_time = time.time()
            y_true, y_pred = model.predict(rdd_data)
            elapsed_time[subset] += time.time() - reference_time
            #
            print(confusion_matrix(y_true, y_pred))
            print(classification_report(y_true, y_pred))
            # Save results in text and graphically represented confusion matrices
            filename_prefix = f'kde-classification-results-bw-%.3f' % band_width
            save_results(f'{results_dir}-{subset}', filename_prefix, y_true, y_pred, elapsed_time = elapsed_time[subset])
        #
        spark_context.stop()
