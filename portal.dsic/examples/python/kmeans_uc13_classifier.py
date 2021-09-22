"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://personales.upv.es/jon)
    Version: 1.0
    Date: September 2021
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

    K-Means-based naive classifier 

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
    Usage: spark-submit --master local[4]  python/kmeans_uc13_classifier.py  \
                                                                --dataset data/uc13.csv \
                                                                --codebook models/kmeans_model-uc13-1000.pkl \
                                                                --confusion-matrix models/cluster-distribution-1000.csv 2>/dev/null
    """

    verbose = 0
    dataset_filename = 'data/uc13.csv'
    codebook_filename = 'models/kmeans_model-uc13-200.pkl'
    confusion_matrix_filename = 'models/cluster-distribution-200.csv'
    spark_context = None
    slices = 8
    batch_size = 100
                                                   
    for i in range(len(sys.argv)):
        if sys.argv[i] == "--dataset":
            dataset_filename = sys.argv[i + 1]
        elif sys.argv[i] == "--verbosity":
            verbose = int(sys.argv[i + 1])
        elif sys.argv[i] == "--num-slices":
            slices = int(sys.argv[i + 1])
        elif sys.argv[i] == "--batch-size":
            batch_size = int(sys.argv[i + 1])
        elif sys.argv[i] == "--codebook":
            codebook_filename = sys.argv[i + 1]
        elif sys.argv[i] == "--confusion-matrix":
            confusion_matrix_filename = sys.argv[i + 1]

    spark_context = SparkContext(appName = "K-Means-based naive classifier")


    # Load the codebook
    with open(codebook_filename, 'rb') as f:
        centers = pickle.load(f)
        f.close()

    kmeans = KMeans()
    kmeans.n_clusters = len(centers)
    kmeans.cluster_centers_ = numpy.array(centers)
    print(f'loaded the codebook of {kmeans.n_clusters} clusters')

    # Load the confusion matrix
    confusion_matrix = None
    with open(confusion_matrix_filename, 'rt') as f:
        confusion_matrix = list()
        for line in f:
            confusion_matrix.append([float(x) for x in line.split(';')])
        f.close()
        confusion_matrix = numpy.array(confusion_matrix)

    # Compute the conditional probabilities
    conditional_probabilities = confusion_matrix.copy()
    # normalize per row to compensate the unbalance of target classes
    for i in range(confusion_matrix.shape[0]):
        conditional_probabilities[i, :] = confusion_matrix[i, :] / sum(confusion_matrix[i, :])
    # normalize per column to use this K-Means-based naive classifier
    for j in range(conditional_probabilities.shape[1]):
        conditional_probabilities[:, j] = conditional_probabilities[:, j] / sum(conditional_probabilities[:, j])

    # Load and parse the data
    csv_lines = spark_context.textFile(dataset_filename)
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

    def classify_sample(t):
        k = kmeans.predict([t[1]])
        k = conditional_probabilities[:, k[0]].argmax()
        return (t[0], k)
        
    data = data.map(classify_sample)

    def compute_assigments(t):
        x = numpy.zeros(10, dtype = int) # 10 must be parametrized
        x[t[1]] = 1
        return (t[0], x)

    data = data.map(compute_assigments)

    accum_matrix = data.reduceByKey(lambda x, y: x + y).collect()
    
    spark_context.stop()

    matrix = numpy.zeros([len(accum_matrix), len(accum_matrix)], dtype = int)
    for row in accum_matrix:
        l = row[0]
        x = row[1]
        matrix[l, :] = x

    f = open(f'results/classification-results-{kmeans.n_clusters}.txt', 'wt')
    for l in range(len(matrix)):
        print(" ".join("{:20d}".format(v) for v in matrix[l]), file = f)
    f.close()
