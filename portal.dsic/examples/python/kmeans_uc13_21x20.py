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

from machine_learning import KMeans as JonKMeans

from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans, KMeansModel



def load_conditional_probabilities(counter_pairs_filename):
    counter_pairs = None
    with open(counter_pairs_filename, 'rt') as f:
        counter_pairs = list()
        for line in f:
            counter_pairs.append([float(x) for x in line.split(';')])
        f.close()
        counter_pairs = numpy.array(counter_pairs)

    # Compute the conditional probabilities
    conditional_probabilities = counter_pairs.copy()
    for i in range(counter_pairs.shape[0]):
        conditional_probabilities[i, :] = counter_pairs[i, :] / sum(counter_pairs[i, :])
            
    # Compute the a priori probabilities of target classes
    target_class_a_priori_probabilities = counter_pairs.sum(axis = 1) / counter_pairs.sum()

    return conditional_probabilities, target_class_a_priori_probabilities



if __name__ == "__main__":
    sc = SparkContext(appName = "kmeans-uc13-21x20")  # SparkContext

    debug = 0
    num_channels = 21
    base_dir = 'uc13-21x20'
    patient = 'chb01'
    min_num_clusters = 10
    max_num_clusters = 150
    num_partitions = 80
    do_train = False
    do_classify = False

    for i in range(1, len(sys.argv)):
        if   sys.argv[i] == '--patient'         : patient = sys.argv[i + 1]
        elif sys.argv[i] == '--models-dir'      : models_dir = sys.argv[i + 1]
        elif sys.argv[i] == '--n-partitions'    : num_partitions = int(sys.argv[i + 1])
        elif sys.argv[i] == '--min-num-clusters': min_num_clusters = int(sys.argv[i + 1])
        elif sys.argv[i] == '--max-num-clusters': max_num_clusters = int(sys.argv[i + 1])
        elif sys.argv[i] == '--train'           : do_train = True
        elif sys.argv[i] == '--classify'        : do_classify = True

    models_dir  = f'{base_dir}/{patient}/models'
    log_dir     = f'{base_dir}/{patient}/log'
    results_dir = f'{base_dir}/{patient}/results'

    os.makedirs(models_dir, exist_ok = True)
    os.makedirs(log_dir,    exist_ok = True)

    # Prepares the list of codebook sizes to explore
    list_of_num_clusters = list()
    num_clusters = min_num_clusters
    delta_num_clusters = 10
    while num_clusters <= max_num_clusters:
        list_of_num_clusters.append(num_clusters)
        if num_clusters >= 500:
            delta_num_clusters = 100
        elif num_clusters >= 150:
            delta_num_clusters = 50
        #
        num_clusters += delta_num_clusters
    #
    print(list_of_num_clusters)


    def csv_line_to_patient_label_and_sample(line):
        parts = line.split(';')
        patient = parts[0]
        label = int(parts[1])
        x = numpy.array([float(x) for x in parts[2:]])
        #x = x.reshape(21, -1)
        #x = x - x.min(axis = 1).reshape(-1, 1)
        #x = x / x.max(axis = 1).reshape(-1, 1)
        return (patient, label, x.flatten())

    # Loads and repartitions the data
    data_rdds = dict()
    for subset in ['train', 'test']:
        csv_lines = sc.textFile(f'data/uc13/uc13-{patient}-21x20-{subset}.csv')
        csv_lines = csv_lines.repartition(num_partitions)
        data = csv_lines.map(csv_line_to_patient_label_and_sample)
        data_rdds[subset] = data
        data_rdds[subset].persist()
        num_samples = data_rdds[subset].count()
        print(f'loaded {num_samples} {subset} samples into {data.getNumPartitions()} partitions')


    if do_train:
        rdd_train = data_rdds['train']
        # removes first and second columns corresponding to patient and label
        samples = rdd_train.map(lambda x: x[2].reshape(21, -1)).flatMap(lambda x: [x[i] for i in range(x.shape[0])])
        samples.persist()
        num_samples = samples.count()
        print(f'working with {num_samples} for training')

        c_E = samples.reduce(lambda x1, x2: x1 + x2) / num_samples # for the Calinski-Harabasz index

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

            codebook_filename = f'{models_dir}/kmeans_model-uc13-{patient}-%04d.pkl' % num_clusters
        
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


            kmeans = JonKMeans()
            kmeans.cluster_centers_ = numpy.array(kmeans_model.centers)
            kmeans.n_clusters = len(kmeans.cluster_centers_)

            def assign_sample_to_cluster(t):
                patient, label, sample = t 
                output = list()
                cluster_assignments = kmeans.predict(sample.reshape(num_channels, -1))
                for j in cluster_assignments:
                    x = numpy.zeros(kmeans.n_clusters)
                    x[j] = 1
                    output.append((label, x)) # for this purpose patient is dropped
                return output
            #
            
            matrix = rdd_train.flatMap(assign_sample_to_cluster).reduceByKey(lambda x, y: x + y).collect()
    
            counters = numpy.zeros([len(matrix), kmeans.n_clusters])
            for row in matrix:
                l = row[0]
                x = row[1]
                counters[l] += x
            #
            f = open(f'{models_dir}/cluster-distribution-%04d.csv' % kmeans.n_clusters, 'wt')
            for l in range(len(counters)):
                print(";".join("{:.0f}".format(v) for v in counters[l]), file = f)
            f.close()

        ####################################################################
        samples.unpersist()
        del samples
    # end if do_train

    if do_classify:
        for num_clusters in list_of_num_clusters:
            codebook_filename = f'{models_dir}/kmeans_model-uc13-{patient}-%04d.pkl' % num_clusters
            with open(codebook_filename, 'rb') as f:
                cluster_centers = pickle.load(f)
                f.close()
            #
            kmeans = JonKMeans()
            kmeans.cluster_centers_ = numpy.array(cluster_centers)
            kmeans.n_clusters = len(kmeans.cluster_centers_)
            print(f'loaded codebook for {num_clusters}')

            # Load the confusion matrix
            conditional_probabilities, target_class_a_priori_probabilities = load_conditional_probabilities(f'{models_dir}/cluster-distribution-%04d.csv' % kmeans.n_clusters)

            def classify_sample(t):
                patient, label, sample = t
                cluster_assignment = kmeans.predict(sample.reshape(num_channels, -1))
                probs = numpy.zeros(conditional_probabilities.shape[0]) # one per target class
                for j in cluster_assignment:
                    probs += conditional_probabilities[:, j] 
                #probs *= target_class_a_priori_probabilities
                k = probs.argmax()
                return (patient, label, k)
        
            for subset in ['train', 'test']:
                data = data_rdds[subset]

                classification = data.map(classify_sample)

                y_true_and_pred = classification.collect()
                y_true = numpy.array([x[1] for x in y_true_and_pred])
                y_pred = numpy.array([x[2] for x in y_true_and_pred])

                filename_prefix = f'kmeans-{patient}-21x20-%04d' % kmeans.n_clusters
                save_results(f'{results_dir}.{subset}', filename_prefix, y_true, y_pred)
            #


    # end if do_classify

    for data in data_rdds.values():
        data.unpersist()

    sc.stop()
