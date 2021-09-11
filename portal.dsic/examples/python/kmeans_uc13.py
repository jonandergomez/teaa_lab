#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Code modified from the Spark sample code to adapt it for subject
#
#       14009 "Scalable Machine Learning Techniques"
#
#   Bachelor's degree in Data Science
#   School of Informatics  (http://www.etsinf.upv.es)
#   Technical University of Valencia (http://www.upv.es)
#
#
import sys
import time
import pickle
import math
import numpy

from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans, KMeansModel

if __name__ == "__main__":
    sc = SparkContext(appName = "kmeans-uc13")  # SparkContext

    debug = 0
    filename = 'data/uc13.csv'
    min_num_clusters = 200 # 51 # 2
    max_num_clusters = 500 # 150 # 50
    delta_num_clusters = 50
    num_partitions = 80

    for i in range(1, len(sys.argv)):
        if sys.argv[i] == '--filename':
            filename = sys.argv[i + 1]
        elif sys.argv[i] == '--n-clusters':
            num_clusters = int(sys.argv[i + 1])
        elif sys.argv[i] == '--n-partitions':
            num_partitions = int(sys.argv[i + 1])

    # Load and parse the data
    lines = sc.textFile(filename)
    data = lines.map(lambda line: numpy.array([float(x) for x in line.split(';')]))
    data = data.map(lambda x: x[1:]) # removes the first column corresponding to the label
    x = data.take(1)
    num_samples = data.count()
    dim = x[0].shape[0]

    print(f'loaded {num_samples} {dim}-dimensional samples into {data.getNumPartitions()} partitions')

    data.repartition(num_partitions)

    if debug > 1: print(type(x), x[0].shape)

    print(f'distributed {num_samples} {dim}-dimensional samples into {data.getNumPartitions()} partitions')

    ####################################################################
    # do standard scaling
    x_mean = data.reduce(lambda x1, x2: x1 + x2) / num_samples
    if debug > 0: print('mean:', x_mean)
    x_std = numpy.sqrt(data.map(lambda x: (x - x_mean) ** 2).reduce(lambda x1, x2: x1 + x2) / num_samples)
    if debug > 0: print('std:', x_std)
    data = data.map(lambda x: (x - x_mean) / x_std)
    ####################################################################

    data.persist()

    c_E = data.reduce(lambda x1, x2: x1 + x2) / num_samples # for the Calinski-Harabasz index

    ####################################################################
    # Evaluate clustering by computing Within Set Sum of Squared Errors

    def compute_values_for_metrics(x):
        q = kmeans_model.predict(x)
        c_q = kmeans_model.centers[q]
        d = x - c_q
        sd = sum(d ** 2)
        return q, sd, math.sqrt(sd), numpy.outer(d, d)
    ####################################################################
    
    for num_clusters in range(min_num_clusters, max_num_clusters + 1, delta_num_clusters):

        # Build the model (cluster the data)
        starting_time = time.time()
        kmeans_model = KMeans.train(data, k = num_clusters, maxIterations = 100, initializationMode = "kmeans||", initializationSteps = 5, epsilon = 1.0e-4)
        ending_time = time.time()
        print('processing time lapse for', num_clusters, 'clusters', ending_time - starting_time, 'seconds')


        starting_time = time.time()
        values_for_metrics = data.map(lambda x: compute_values_for_metrics(x))
        values_for_metrics.persist()
        WSSSE = values_for_metrics.map(lambda x: x[1]).reduce(lambda x, y: x + y)
        W_k   = values_for_metrics.map(lambda x: x[3]).reduce(lambda x, y: x + y)
        n_q   = values_for_metrics.map(lambda x: (x[0], 1)).countByKey()
        S     = values_for_metrics.map(lambda x: (x[0], x[2])).reduceByKey(lambda c1, c2: c1 + c2).collect()
        values_for_metrics.unpersist()
        ending_time = time.time()
        print('computing metrics time lapse for', num_clusters, 'clusters', ending_time - starting_time, 'seconds')

        #WSSSE = data.map(lambda point: error(point)).reduce(lambda x, y: x + y)
        #WSSSE = kmeans_model.computeCost(data)
        print(f'Within Set Sum of Squared Error (WSSSE) is {WSSSE} and normalized per sample is {WSSSE / num_samples}')

        # Calinski-Harabasz index
        if debug > 1: print(n_q)
        B_k = sum([n_q[q] * numpy.outer(kmeans_model.centers[q] - c_E, (kmeans_model.centers[q] - c_E)) for q in range(num_clusters)])
        calinski_harabasz_index = (sum(numpy.diag(B_k)) / sum(numpy.diag(W_k))) * ((num_samples - num_clusters) / (num_clusters - 1))
        print(f'Calinski Harabasz index for {num_clusters} clusters is {calinski_harabasz_index}  and normalized per sample is {calinski_harabasz_index / num_samples}')

        #S = data.map(lambda x : compute_davies_bouldin_s_i(x)).reduceByKey(lambda c1, c2: c1 + c2).collect()
        S.sort(key = lambda x : x[0])
        if debug > 2: print(S)
        S = [s[1] for s in S]
        if debug > 2: print(S)
        for q in range(num_clusters):
            S[q] /= n_q[q]
        if debug > 2: print(S)
        R = numpy.zeros([num_clusters, num_clusters])
        davies_bouldin_index = 0
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                R[i, j] = R[j, i] = (S[i] + S[j]) / max(1.0e-3, math.sqrt(sum((kmeans_model.centers[i] - kmeans_model.centers[j]) ** 2))) # set lower bound to 1.0e-3 to avoid zero divisions

            j_max = numpy.argmax(R[i, :])
            assert j_max != i
            davies_bouldin_index += R[i, j_max]
        davies_bouldin_index /= num_clusters
        print(f'Davies Bouldin index for {num_clusters} clusters is {davies_bouldin_index}')

        if debug > 1: print(len(kmeans_model.centers), kmeans_model.centers[0].shape)

        with open('data/kmeans_model-uc13-%03d.pkl' % num_clusters, 'wb') as f:
            pickle.dump(kmeans_model.centers, f)
            f.close()

    ####################################################################
    data.unpersist()
    sc.stop()
