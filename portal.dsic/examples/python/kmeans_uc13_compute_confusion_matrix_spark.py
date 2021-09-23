"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://personales.upv.es/jon)
    Version: 1.0
    Date: September 2021
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

    Computing the confusionn matrix for a K-Means-based naive classifier 

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

    verbose = 0
    dataset_filename = 'data/uc13.csv'
    codebook_filename = 'models/kmeans_model-uc13-200.pkl'
    spark_context = None
    num_partitions = 40
    batch_size = 100
                                                   
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


    def csv_line_to_label_and_sample(line):
        parts = line.split(';')
        return (int(parts[0]), numpy.array([float(x) for x in parts[1:]]))

    data = csv_lines.map(csv_line_to_label_and_sample)

    ####################################################################
    # do standard scaling
    x_mean = data.map(lambda x: x[1]).reduce(lambda x1, x2: x1 + x2)
    x_mean /= num_samples
    print(x_mean.shape)
    x_std = data.map(lambda x: x[1]).map(lambda x: (x - x_mean) ** 2).reduce(lambda x1, x2: x1 + x2)
    x_std = numpy.sqrt(x_std / num_samples)
    print(x_std.shape)
    data = data.map(lambda x: (x[0], (x[1] - x_mean) / x_std))
    ####################################################################

    def assign_sample_to_cluster(t):
        k = kmeans.predict([t[1]])
        x = numpy.zeros(kmeans.n_clusters)
        x[k[0]] = 1
        return (t[0], x)
        
    data = data.map(assign_sample_to_cluster)

    matrix = data.reduceByKey(lambda x, y: x + y).collect()
    
    spark_context.stop()

    counters = numpy.zeros([len(matrix), kmeans.n_clusters])
    for row in matrix:
        l = row[0]
        x = row[1]
        counters[l] += x

    f = open('models/cluster-distribution-%03d.csv' % kmeans.n_clusters, 'wt')
    for l in range(len(counters)):
        print(";".join("{:f}".format(v) for v in counters[l]), file = f)
    f.close()
