"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://personales.upv.es/jon)
    Version: 1.0
    Date: September 2021
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

    Example of how to run the original K-Means from a CSV file stored in the HDFS

    run with this command:
        hdfs dfs -cat  data/uc13.csv | python python/kmeans_uc13_original.py
"""

import os
import sys
import time
import numpy
import pickle

from machine_learning import KMeans

num_target_classes = 10
num_clusters = 1000

t0 = time.time()


kmeans = KMeans(n_clusters = num_clusters, modality = 'original-k-Means', verbosity = 1)

not_initialized = True
labels = list()
samples = list()
i = 0
for line in sys.stdin:
    parts = line.split(sep = ';')
    labels.append(int(parts[0]))
    samples.append([float(x) for x in parts[1:]])
    #
    i += 1
    if i % 5000 == 0:
        samples = numpy.array(samples)
        labels  = numpy.array(labels, dtype = int)
        if not_initialized:
            kmeans.original_k_means_init(samples)
        else:
            kmeans.original_k_means_iteration(samples)

        print('execution time at sample', i, 'is', time.time() - t0, 'seconds')
        samples = list()
        labels = list()

if len(samples) > 0:
    samples = numpy.array(samples)
    labels  = numpy.array(labels, dtype = int)
    kmeans.original_k_means_iteration(samples)

print('execution time:', time.time() - t0, 'seconds')

with open('models/kmeans_model-uc13-%03d.pkl' % num_clusters, 'wb') as f:
    pickle.load(f, kmeans.cluster_centers_)
    f.close()
