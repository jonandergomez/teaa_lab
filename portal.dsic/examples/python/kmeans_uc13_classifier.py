"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://personales.upv.es/jon)
    Version: 1.0
    Date: September 2021
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

    Simple classifier by associating clusters from K-Means to real classes
"""

import os
import sys
import time
import numpy
import pickle

from machine_learning import KMeans

num_target_classes = 10
num_clusters = 5

t0 = time.time()

with open('models/kmeans_model-uc13-%03d.pkl' % num_clusters, 'rb') as f:
    centers = pickle.load(f)
    f.close()

kmeans = KMeans()
kmeans.n_clusters = len(centers)
kmeans.cluster_centers_ = numpy.array(centers)

#print(kmeans.n_clusters, kmeans.cluster_centers_.shape)
#sys.exit(0)

counters = list()
for l in range(num_target_classes):
    counters.append([0] * num_clusters)

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
        k = kmeans.predict(samples)
        for j in range(len(k)):
            counters[labels[j]][k[j]] += 1
        print('execution time at sample', i, 'is', time.time() - t0, 'seconds')
        labels = list()
        samples = list()

if len(samples) > 0:
    samples = numpy.array(samples)
    labels  = numpy.array(labels, dtype = int)
    k = kmeans.predict(samples)
    for j in range(len(k)):
        counters[labels[j]][k[j]] += 1

print('execution time:', time.time() - t0, 'seconds')

f = open(f'cluster-distribution-{num_clusters}.csv', 'wt')
for l in range(len(counters)):
    print(";".join("{:d}".format(v) for v in counters[l]), file = f)
f.close()
