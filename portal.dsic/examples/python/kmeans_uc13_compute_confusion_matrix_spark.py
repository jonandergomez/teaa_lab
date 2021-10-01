"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://personales.upv.es/jon)
    Version: 1.0
    Date: September 2021
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

    Computing the confusion matrix for a K-Means-based naive classifier 

"""

import os
import sys
import numpy
import pickle

from machine_learning import KMeans

try:
    from pyspark import SparkContext
except:
    pass


if __name__ == "__main__":

    """
    Usage: spark-submit --master local[4]  python/kmeans_uc13_compute_confusion_matrix_spark.py  \
                                                                --dataset data/uc13.csv \
                                                                --codebook models/kmeans_model-uc13-1000.pkl  2>/dev/null
    """

    label_mapping = [i for i in range(10)]

    verbose = 0
    dataset_filename = 'data/uc13-train.csv'
    codebook_filename = None
    spark_context = None
    num_partitions = 40
    batch_size = 100
    num_channels = 21
    models_dir = 'models'
    log_dir = 'log'
    do_reshape = True
    do_standard_scaling = True
                                                   
    for i in range(len(sys.argv)):
        if sys.argv[i] == "--dataset":
            dataset_filename = sys.argv[i + 1]
        elif sys.argv[i] == "--verbosity":
            verbose = int(sys.argv[i + 1])
        elif sys.argv[i] == "--num-partitions":
            num_partitions = int(sys.argv[i + 1])
        elif sys.argv[i] == "--batch-size":
            batch_size = int(sys.argv[i + 1])
        elif sys.argv[i] == "--codebook":
            codebook_filename = sys.argv[i + 1]
        elif sys.argv[i] == "--reduce-labels":
            label_mapping[3] = 0
            label_mapping[4] = 0
            label_mapping[5] = 0
            label_mapping[6] = 1
            label_mapping[7] = 0
            label_mapping[8] = 0
            label_mapping[9] = 0
        elif sys.argv[i] == "--from-pca":
            models_dir = 'models.pca'
            log_dir = 'log.pca'
            do_reshape = False
            do_standard_scaling = False

    spark_context = SparkContext(appName = "K-Means compute confusion matrix")


    with open(codebook_filename, 'rb') as f:
        centers = pickle.load(f)
        f.close()

    kmeans = KMeans()
    kmeans.n_clusters = len(centers)
    kmeans.cluster_centers_ = numpy.array(centers)
    print(f'loaded the codebook of {kmeans.n_clusters} clusters')


    # Load and parse the data
    csv_lines = spark_context.textFile(dataset_filename)
    #csv_lines = csv_lines.coalesce(20)
    csv_lines = csv_lines.repartition(num_partitions)
    print("file(s) loaded ")
    csv_lines.persist()
    num_samples = csv_lines.count()
    print("loaded %d samples distributed in %d partitions" % (num_samples, csv_lines.getNumPartitions()))
    csv_lines.unpersist()


    def csv_line_to_patient_label_and_sample_reshape(line):
        parts = line.split(';')
        return (parts[0], label_mapping[int(parts[1])], numpy.array([float(x) for x in parts[2:]]).reshape(num_channels, 14))

    def csv_line_to_patient_label_and_sample(line):
        parts = line.split(';')
        return (parts[0], label_mapping[int(parts[1])], numpy.array([float(x) for x in parts[2:]]))

    if do_reshape:
        data = csv_lines.map(csv_line_to_patient_label_and_sample_reshape)
    else:
        data = csv_lines.map(csv_line_to_patient_label_and_sample)

    if do_standard_scaling:
        ####################################################################
        # begin: do standard scaling 
        ####################################################################
        statistics_filename = f'{models_dir}/mean_and_std.pkl'
        if os.path.exists(statistics_filename):
            with open(statistics_filename, 'rb') as f:
                stats = pickle.load(f)
                f.close()
            x_mean, x_std = stats
        else:
            x_mean = data.map(lambda x: x[2]).reduce(lambda x1, x2: x1 + x2)
            x_mean = x_mean.sum(axis = 0) # merging all the channels altogether, i.e. sum(axis = 0)
            x_mean /= (num_samples * num_channels)
            print(x_mean.shape) # should be (14,)
            x_std = data.map(lambda x: x[2]).map(lambda x: (x - x_mean) ** 2).reduce(lambda x1, x2: x1 + x2)
            x_std = x_std.sum(axis = 0) # merging all the channels altogether, i.e. sum(axis = 0)
            x_std = numpy.sqrt(x_std / (num_samples * num_channels))
            print(x_std.shape) # should be (14,)
            with open(statistics_filename, 'wb') as f:
                pickle.dump([x_mean, x_std], f)
                f.close()
        #
        data = data.map(lambda x: (x[0], x[1], (x[2] - x_mean) / x_std))
        ####################################################################
        # end: do standard scaling 
        ####################################################################

    if do_reshape:
        def assign_sample_to_cluster(t):
            patient, label, sample = t 
            output = list()
            cluster_assignments = kmeans.predict(sample)
            for j in cluster_assignments:
                x = numpy.zeros(kmeans.n_clusters)
                x[j] = 1
                output.append((label, x)) # for this purpose patient is dropped
            return output
            
        data = data.flatMap(assign_sample_to_cluster)
    else:
        def assign_sample_to_cluster(t):
            patient, label, sample = t 
            cluster_assignments = kmeans.predict([sample])
            x = numpy.zeros(kmeans.n_clusters)
            j = cluster_assignments[0]
            x[j] = 1
            return (label, x) # for this purpose patient is dropped

        data = data.map(assign_sample_to_cluster)
    #

    matrix = data.reduceByKey(lambda x, y: x + y).collect()
    
    spark_context.stop()

    counters = numpy.zeros([len(matrix), kmeans.n_clusters])
    for row in matrix:
        l = row[0]
        x = row[1]
        counters[l] += x

    f = open(f'{models_dir}/cluster-distribution-%03d.csv' % kmeans.n_clusters, 'wt')
    for l in range(len(counters)):
        print(";".join("{:.0f}".format(v) for v in counters[l]), file = f)
    f.close()
