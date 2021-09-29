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
import os
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
    filename = 'data/uc13-train.csv'
    num_channels = 21
    do_reshape = True
    do_standard_scaling = True
    models_dir = 'models'
    log_dir = 'log'

    list_of_num_clusters = list()
    #for k in range(   2,  150,   1): list_of_num_clusters.append(k) # comment this line to skip these clustering sizes
    for k in range(  10,  150,  10): list_of_num_clusters.append(k) # comment this line to skip these clustering sizes
    for k in range( 150,  500,  50): list_of_num_clusters.append(k) # comment this line to skip these clustering sizes
    for k in range( 500, 1000, 100): list_of_num_clusters.append(k) # comment this line to skip these clustering sizes
    list_of_num_clusters.append(1000)
    #list_of_num_clusters.append(60)

    num_partitions = 80

    for i in range(1, len(sys.argv)):
        if sys.argv[i] == '--filename':
            filename = sys.argv[i + 1]
        elif sys.argv[i] == '--models-dir':
            models_dir = sys.argv[i + 1]
        elif sys.argv[i] == '--n-clusters':
            list_of_num_clusters.append(int(sys.argv[i + 1]))
            print(list_of_num_clusters)
        elif sys.argv[i] == '--n-partitions':
            num_partitions = int(sys.argv[i + 1])
        elif sys.argv[i] == '--no-reshape':
            do_reshape = False
        elif sys.argv[i] == '--no-standard-scaling':
            do_standard_scaling = False
        elif sys.argv[i] == '--from-pca':
            do_standard_scaling = False
            do_reshape = False
            models_dir = 'models.pca'
            log_dir = 'log.pca'

    # Load and parse the data
    csv_lines = sc.textFile(filename)
    csv_lines = csv_lines.repartition(num_partitions)


    def csv_line_to_patient_label_and_sample_reshape(line):
        parts = line.split(';')
        return (parts[0], int(parts[1]), numpy.array([float(x) for x in parts[2:]]).reshape(num_channels, 14))

    def csv_line_to_patient_label_and_sample(line):
        parts = line.split(';')
        return (parts[0], int(parts[1]), numpy.array([float(x) for x in parts[2:]]))

    if do_reshape:
        data = csv_lines.map(csv_line_to_patient_label_and_sample_reshape)
    else:
        data = csv_lines.map(csv_line_to_patient_label_and_sample)
    num_samples = data.count()

    print(f'loaded {num_samples} samples into {data.getNumPartitions()} partitions')

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

    # removes first and second columns corresponding to patient and label
    if do_reshape:
        data = data.flatMap(lambda x: [x[2][i] for i in range(x[2].shape[0])])
    else:
        data = data.map(lambda x: x[2])

    data.persist()

    c_E = data.reduce(lambda x1, x2: x1 + x2) / num_samples # for the Calinski-Harabasz index

    ####################################################################
    # Evaluate clustering by computing Within Set Sum of Squared Errors

    def compute_values_for_metrics(x):
        q = kmeans_model.predict(x)
        c_q = kmeans_model.centers[q]
        d = x - c_q
        d2 = d ** 2
        sd = sum(d2)
        return q, sd, math.sqrt(sd), d2 # numpy.diag(numpy.outer(d, d)) # use diag of outer product to save memory
    ####################################################################
    
    for num_clusters in list_of_num_clusters:

        codebook_filename = f'{models_dir}/kmeans_model-uc13-%03d.pkl' % num_clusters
        
        if os.path.exists(codebook_filename):
            with open(codebook_filename, 'rb') as f:
                cluster_centers = pickle.load(f)
                f.close()
            
            kmeans_model = KMeansModel(cluster_centers)
            print(f'loaded codebook for {num_clusters}')

        else: # Build the model (cluster the data)
            starting_time = time.time()
            #kmeans_model = KMeans.train(data, k = num_clusters, maxIterations = 100, initializationMode = "kmeans||", initializationSteps = 5, epsilon = 1.0e-4)
            kmeans_model = KMeans.train(data, k = num_clusters, maxIterations = 2000, initializationMode = "kmeans||", initializationSteps = 5, epsilon = 1.0e-9)
            ending_time = time.time()
            print('processing time lapse for', num_clusters, 'clusters', ending_time - starting_time, 'seconds')

            if debug > 1: print(len(kmeans_model.centers), kmeans_model.centers[0].shape)

            with open(codebook_filename, 'wb') as f:
                pickle.dump(kmeans_model.centers, f)
                f.close()


        starting_time = time.time()
        values_for_metrics = data.map(compute_values_for_metrics)
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
        #calinski_harabasz_index = (sum(numpy.diag(B_k)) / sum(numpy.diag(W_k))) * ((num_samples - num_clusters) / (num_clusters - 1))
        calinski_harabasz_index = (sum(numpy.diag(B_k)) / sum(W_k)) * ((num_samples - num_clusters) / (num_clusters - 1))
        print(f'Calinski Harabasz index for {num_clusters} clusters is {calinski_harabasz_index}  and normalized per sample is {calinski_harabasz_index / num_samples}')

        # Davies-Bouldin index
        S.sort(key = lambda x : x[0]) # sort to ensure S[i] corresponds to cluster i
        if debug > 2: print(S)
        S = [s[1] for s in S] # use only the accumulated 'cluster radius' and drop the cluster index
        if debug > 2: print(S)
        for q in range(num_clusters):
            S[q] /= n_q[q] # compute the mean to get the 'cluster radius'
        if debug > 2: print(S)
        R = numpy.zeros([num_clusters, num_clusters]) # prepare the matrix R
        davies_bouldin_index = 0
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                d_i_j = math.sqrt(sum((kmeans_model.centers[i] - kmeans_model.centers[j]) ** 2)) # distance between cluster centers
                R[i, j] = R[j, i] = (S[i] + S[j]) / max(1.0e-3, d_i_j) # set lower bound to 1.0e-3 to avoid zero divisions

            j_max = numpy.argmax(R[i, :])
            assert j_max != i
            davies_bouldin_index += R[i, j_max]
        davies_bouldin_index /= num_clusters
        print(f'Davies Bouldin index for {num_clusters} clusters is {davies_bouldin_index}')

        # save KPIs to measure the quality of the clusterings
        with open(f'{log_dir}/kmeans-kpis.txt', 'at') as f:
            print(f'{num_clusters}  {WSSSE / num_samples}  {calinski_harabasz_index}  {davies_bouldin_index}', file = f)
            f.close()

    ####################################################################
    data.unpersist()
    sc.stop()
