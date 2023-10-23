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

from utils_for_results import save_results
from eeg_load_data import load_csv_from_uc13

use_spark = True
try:
    from pyspark import SparkContext
    from pyspark.mllib.clustering import KMeans, KMeansModel
except:
    use_spark = False


# --------------------------------------------------------------------------------
def load_conditional_probabilities(counter_pairs_filename):
    counter_pairs = None
    with open(counter_pairs_filename, 'rt') as f:
        counter_pairs = list()
        for line in f:
            counter_pairs.append([float(x) for x in line.split(';')])
        f.close()
        counter_pairs = numpy.array(counter_pairs)

    # Compute the conditional probabilities
    _temp_ = counter_pairs / numpy.maximum(1.0, counter_pairs.sum(axis = 0).reshape(1, -1))
    conditional_probabilities = _temp_ / numpy.maximum(1, _temp_.sum(axis = 1).reshape(-1, 1))
            
    # Compute the a priori probabilities of target classes
    target_class_a_priori_probabilities = counter_pairs.sum(axis = 1) / counter_pairs.sum()

    return conditional_probabilities, target_class_a_priori_probabilities
# --------------------------------------------------------------------------------



if __name__ == "__main__":

    spark = SparkContext(appName = "kmeans-uc13-21x14")  # SparkContext

    hdfs_url = 'hdfs://teaa-master-ubuntu22:8020'

    debug = 0
    num_channels = 21
    base_dir = '.'
    patient = 'chb02'
    min_num_clusters = 300
    max_num_clusters = 300 # 1000
    delta_num_clusters = 100
    num_partitions = 60
    do_train = False
    do_classify = False
    do_binary_classification = False

    for i in range(1, len(sys.argv)):
        if   sys.argv[i] == '--patient'                 : patient = sys.argv[i + 1]
        elif sys.argv[i] == '--models-dir'              : models_dir = sys.argv[i + 1]
        elif sys.argv[i] == '--n-partitions'            : num_partitions = int(sys.argv[i + 1])
        elif sys.argv[i] == '--min-num-clusters'        : min_num_clusters = int(sys.argv[i + 1])
        elif sys.argv[i] == '--max-num-clusters'        : max_num_clusters = int(sys.argv[i + 1])
        elif sys.argv[i] == '--delta-num-clusters'      : delta_num_clusters = int(sys.argv[i + 1])
        elif sys.argv[i] == '--train'                   : do_train = True
        elif sys.argv[i] == '--classify'                : do_classify = True
        elif sys.argv[i] == '--do-binary-classification': do_binary_classification = True

    models_dir  = f'{base_dir}/models/uc13/kmeans'
    log_dir     = f'{base_dir}/logs/uc13/kmeans'
    results_dir = f'{base_dir}/results/uc13/kmeans/{patient}'

    task = 'binary-classification' if do_binary_classification else 'multi-class-classification'

    os.makedirs(log_dir,     exist_ok = True)
    os.makedirs(models_dir,  exist_ok = True)
    os.makedirs(results_dir, exist_ok = True)

    # Prepares the list of codebook sizes to explore
    list_of_num_clusters = [n for n in range(min_num_clusters, max_num_clusters + 1, delta_num_clusters)]
    print(list_of_num_clusters)

    if patient == 'ALL':
        train_filenames = [f'{hdfs_url}/data/uc13/21x14/uc13-chb{i:02d}-21x14-time-to-seizure.csv' for i in range(1,17)]
        test_filenames  = [f'{hdfs_url}/data/uc13/21x14/uc13-chb{i:02d}-21x14-time-to-seizure.csv' for i in range(17,25)]
    else:
        train_filenames = [f'{hdfs_url}/data/uc13/21x14/uc13-{patient}-21x14-time-to-seizure-train.csv']
        test_filenames  = [f'{hdfs_url}/data/uc13/21x14/uc13-{patient}-21x14-time-to-seizure-test.csv']

    # Loads and repartitions the data
    rdd_train = load_csv_from_uc13(spark, train_filenames, num_partitions, do_binary_classification = do_binary_classification)
    rdd_test  = load_csv_from_uc13(spark,  test_filenames, num_partitions, do_binary_classification = do_binary_classification)

    # BEGIN: Perform the standard scalation
    mean = rdd_train.map(lambda sample: sample[3]).reduce(lambda x, y: x + y) / rdd_train.count()
    variance = rdd_train.map(lambda sample: (sample[3] - mean)**2).reduce(lambda x, y: x + y) / rdd_train.count()
    sigma = numpy.sqrt(variance)
    #
    rdd_train = rdd_train.map(lambda sample: (sample[0], sample[1], sample[2], (sample[3] - mean) / sigma))
    rdd_test  =  rdd_test.map(lambda sample: (sample[0], sample[1], sample[2], (sample[3] - mean) / sigma))
    # END: Perform the standard scalation

    #print(rdd_train.first())
    #print(rdd_test.first())
    print(rdd_train.count(), rdd_train.getNumPartitions(), rdd_test.count(), rdd_test.getNumPartitions())


    if do_binary_classification:
        labels = [0, 1]
    else:
        l1 = rdd_train.map(lambda sample: (sample[2], 1)).reduceByKey(lambda x, y: x + y).collect()
        l2 = rdd_test.map(lambda sample: (sample[2], 1)).reduceByKey(lambda x, y: x + y).collect()
        labels = [x[0] for x in (l1 + l2)]
        labels = list(numpy.unique(labels))

    print(labels)

    if do_train:
        # removes patient id, tts and label
        samples = rdd_train.map(lambda x: x[3])
        samples.persist()
        num_samples = samples.count()
        c_E = samples.reduce(lambda x, y: x + y) / samples.count() # for the Calinski-Harabasz index
        print(f'working with {num_samples} for training')


        ####################################################################
        # Evaluate clustering by computing Within Set Sum of Squared Error / Calinksi-Harabasz / Davies-Bouldin
        def compute_values_for_metrics(x):
            q = kmeans_model.predict(x)
            c_q = kmeans_model.centers[q]
            d = x - c_q
            d2 = d ** 2
            sd = sum(d2)
            return q, sd, math.sqrt(sd), d2 # numpy.diag(numpy.outer(d, d)) # use diag of outer product to save memory
        ####################################################################
    
        for num_clusters in list_of_num_clusters:

            codebook_filename = f'{models_dir}/kmeans-21x14-{patient}-{num_clusters:04d}.pkl'
        
            if os.path.exists(codebook_filename):
                with open(codebook_filename, 'rb') as f:
                    cluster_centers = pickle.load(f)
                    f.close()
                
                kmeans_model = KMeansModel(cluster_centers)
                print(f'loaded codebook for {num_clusters}')

            else: # Build the model (cluster the data)
                starting_time = time.time()
                #kmeans_model = KMeans.train(samples, k = num_clusters, maxIterations = 100, initializationMode = "kmeans||", initializationSteps = 5, epsilon = 1.0e-4)
                kmeans_model = KMeans.train(samples, k = num_clusters, maxIterations = 2000, initializationMode = "kmeans||", initializationSteps = 5, epsilon = 1.0e-9)
                ending_time = time.time()
                print('processing time lapse for', num_clusters, 'clusters', ending_time - starting_time, 'seconds')

                if debug > 1: print(len(kmeans_model.centers), kmeans_model.centers[0].shape)
                with open(codebook_filename, 'wb') as f:
                    pickle.dump(kmeans_model.centers, f)
                    f.close()

            starting_time = time.time()
            values_for_metrics = samples.map(compute_values_for_metrics)
            values_for_metrics.persist()
            WSSSE = values_for_metrics.map(lambda x: x[1]).reduce(lambda x, y: x + y)
            W_k   = values_for_metrics.map(lambda x: x[3]).reduce(lambda x, y: x + y)
            n_q   = values_for_metrics.map(lambda x: (x[0], 1)).countByKey()
            S     = values_for_metrics.map(lambda x: (x[0], x[2])).reduceByKey(lambda c1, c2: c1 + c2).collect()
            values_for_metrics.unpersist()
            ending_time = time.time()

            print('computing metrics time lapse for', num_clusters, 'clusters', ending_time - starting_time, 'seconds')

            print(f'Within Set Sum of Squared Error (WSSSE) is {WSSSE} and normalized per sample is {WSSSE / num_samples}')

            # Calinski-Harabasz index
            if debug > 1: print(n_q)
            B_k = sum([n_q[q] * numpy.outer(kmeans_model.centers[q] - c_E, kmeans_model.centers[q] - c_E) for q in range(num_clusters)])
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
            _centers_ = kmeans_model.centers
            #
            for i in range(num_clusters):
                for j in range(i + 1, num_clusters):
                    d_i_j = math.sqrt(sum((_centers_[i] - _centers_[j]) ** 2)) # distance between cluster centers
                    R[i, j] = R[j, i] = (S[i] + S[j]) / max(1.0e-3, d_i_j) # set lower bound to 1.0e-3 to avoid zero divisions
                #
                j_max = numpy.argmax(R[i, :])
                assert j_max != i
                davies_bouldin_index += R[i, j_max]
            #
            davies_bouldin_index /= num_clusters
            print(f'Davies Bouldin index for {num_clusters} clusters is {davies_bouldin_index}')

            # save KPIs to measure the quality of the clusterings
            with open(f'{log_dir}/kmeans-kpis-{patient}.txt', 'at') as f:
                print(f'{num_clusters}  {WSSSE / num_samples}  {calinski_harabasz_index}  {davies_bouldin_index}', file = f)
                f.close()

            def assign_sample_to_cluster(t):
                patient, tts, label, sample = t 
                x = numpy.zeros(num_clusters)
                j = kmeans_model.predict(sample)
                x[j] = 1
                return (label, x)
            #
            
            matrix = rdd_train.map(assign_sample_to_cluster).reduceByKey(lambda x, y: x + y).collect()
            counters = numpy.zeros([len(matrix), num_clusters])
            for row in matrix:
                l = row[0]
                x = row[1]
                counters[l] += x
            #
            del matrix
            #
            f = open(f'{models_dir}/cluster-distribution-{patient}-{task}-{num_clusters:04d}.csv', 'wt')
            for l in range(len(counters)):
                print(";".join("{:.0f}".format(v) for v in counters[l]), file = f)
            f.close()

        ####################################################################
        samples.unpersist()
        del samples
    # end if do_train

    if do_classify:
        for num_clusters in list_of_num_clusters:
            codebook_filename = f'{models_dir}/kmeans-21x14-{patient}-{num_clusters:04d}.pkl'
            with open(codebook_filename, 'rb') as f:
                cluster_centers = pickle.load(f)
                f.close()
            #
            kmeans_model = KMeansModel(cluster_centers)
            print(f'loaded codebook for {num_clusters}')

            # Load the confusion matrix
            conditional_probabilities, target_class_a_priori_probabilities = load_conditional_probabilities(f'{models_dir}/cluster-distribution-{patient}-{task}-{num_clusters:04d}.csv')

            def classify_sample(t):
                patient, tts, label, sample = t
                j = kmeans_model.predict(sample)
                probs = conditional_probabilities[:, j]
                k = probs.argmax()
                return (patient, label, k)
        
            data_rdds = {'train' : rdd_train, 'test' : rdd_test}

            for subset in ['train', 'test']:
                data = data_rdds[subset]
                classification = data.map(classify_sample)
                y_true_and_pred = classification.collect()
                y_true = numpy.array([x[1] for x in y_true_and_pred])
                y_pred = numpy.array([x[2] for x in y_true_and_pred])

                filename_prefix = f'kmeans-21x14-{patient}-{task}-{num_clusters:04d}'
                save_results(f'{results_dir}/{subset}', filename_prefix, y_true, y_pred, labels = labels)
            #

            for data in data_rdds.values(): data.unpersist()
    # end if do_classify

    spark.stop()
