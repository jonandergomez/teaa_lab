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

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans, KMeansModel

from utils_for_results import save_results

def load_mnist():
#
    home = os.getenv('HOME')
    filename = None
    if home is not None:
        filename = home + '/scikit_learn_data/openml/openml.org/mnist_784.npz'
    else:
        filename = '/tmp/mnist_784.npz' # This will fail in Windows machines
    #
    if os.path.exists(filename):
        npz = numpy.load(filename, allow_pickle = True)
        X, y = npz['X'], npz['y']
    else:
        X, y = fetch_openml('mnist_784', version = 'active', return_X_y = True)
        numpy.savez(filename, X = X, y = y)
    #
    y = numpy.array([int(_) for _ in y])
    return X, y


if __name__ == "__main__":
    sc = SparkContext(appName = "kmeans-mnist")  # SparkContext

    X, y = load_mnist()
    X /= 255.0
    print(X.shape, y.shape)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    #
    pca = PCA(n_components = 0.95)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)


    debug = 0
    models_dir = 'models.digits'
    log_dir = 'log.digits'

    list_of_num_clusters = list()
    #for k in range(   2,  150,   1): list_of_num_clusters.append(k) # comment this line to skip these clustering sizes
    for k in range(  10,  150,  10): list_of_num_clusters.append(k) # comment this line to skip these clustering sizes
    for k in range( 150,  500,  50): list_of_num_clusters.append(k) # comment this line to skip these clustering sizes
    for k in range( 500, 1000, 100): list_of_num_clusters.append(k) # comment this line to skip these clustering sizes
    list_of_num_clusters.append(1000)

    num_partitions = 60

    for i in range(1, len(sys.argv)):
        if sys.argv[i] == '--models-dir':
            models_dir = sys.argv[i + 1]
        elif sys.argv[i] == '--n-clusters':
            list_of_num_clusters.append(int(sys.argv[i + 1]))
            print(list_of_num_clusters)
        elif sys.argv[i] == '--n-partitions':
            num_partitions = int(sys.argv[i + 1])


    rdd_train = sc.parallelize([x.copy() for x in X_train], numSlices = num_partitions)
    rdd_test  = sc.parallelize([x.copy() for x in X_test ], numSlices = num_partitions)
    num_samples = rdd_train.count()
    print(f'train subset with {num_samples} distributed into {rdd_train.getNumPartitions()} partitions')
    print(f'test  subset with {rdd_test.count()} distributed into {rdd_test.getNumPartitions()} partitions')

    rdd_train.persist()

    c_E = rdd_train.reduce(lambda x1, x2: x1 + x2) / num_samples # for the Calinski-Harabasz index

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

        codebook_filename = f'{models_dir}/kmeans_model-mnist-%03d.pkl' % num_clusters
        
        if os.path.exists(codebook_filename):
            with open(codebook_filename, 'rb') as f:
                cluster_centers = pickle.load(f)
                f.close()
            
            kmeans_model = KMeansModel(cluster_centers)
            print(f'loaded codebook for {num_clusters}')

        else: # Build the model (cluster the data)
            starting_time = time.time()
            kmeans_model = KMeans.train(rdd_train,  k = num_clusters,
                                                    maxIterations = 2000,
                                                    initializationMode = "kmeans||",
                                                    initializationSteps = 5,
                                                    epsilon = 1.0e-9)
            ending_time = time.time()
            print('processing time lapse for', num_clusters, 'clusters', ending_time - starting_time, 'seconds')

            if debug > 1: print(len(kmeans_model.centers), kmeans_model.centers[0].shape)

            with open(codebook_filename, 'wb') as f:
                pickle.dump(kmeans_model.centers, f)
                f.close()


        starting_time = time.time()
        values_for_metrics = rdd_train.map(compute_values_for_metrics)
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

            j_max = numpy.argmax(R[i, :])
            assert j_max != i
            davies_bouldin_index += R[i, j_max]
        davies_bouldin_index /= num_clusters
        print(f'Davies Bouldin index for {num_clusters} clusters is {davies_bouldin_index}')

        # save KPIs to measure the quality of the clusterings
        with open(f'{log_dir}/kmeans-kpis.txt', 'at') as f:
            print(f'{num_clusters}  {WSSSE / num_samples}  {calinski_harabasz_index}  {davies_bouldin_index}', file = f)
            f.close()

        ##################################################################################################################
        # Computation of the counts for the conditional probabilities

        def assign_sample_to_cluster(t):
            label, sample = t 
            j = kmeans_model.predict(sample)
            x = numpy.zeros(num_clusters)
            x[j] = 1
            return (label, x)
        # --------------------------------------------------------------------------------------
        train_data = sc.parallelize([(y, x.copy()) for x, y in zip(X_train, y_train)], numSlices = num_partitions)
        test_data  = sc.parallelize([(y, x.copy()) for x, y in zip(X_test,  y_test)],  numSlices = num_partitions)
        data = train_data.map(assign_sample_to_cluster)
        #
        matrix = data.reduceByKey(lambda x, y: x + y).collect()
        
        counters = numpy.zeros([len(matrix), num_clusters])
        for row in matrix:
            l = row[0]
            x = row[1]
            counters[l] += x

        f = open(f'{models_dir}/cluster-distribution-%03d.csv' % num_clusters, 'wt')
        for l in range(len(counters)):
            print(";".join("{:.0f}".format(v) for v in counters[l]), file = f)
        f.close()

        conditional_probabilities = counters / counters.sum(axis = 1).reshape(-1, 1)
        target_class_a_priori_probabilities = counters.sum(axis = 1) / counters.sum()

        def classify_sample(t):
            label, sample = t
            j = kmeans_model.predict(sample)
            probs = conditional_probabilities[:, j] 
            probs *= target_class_a_priori_probabilities
            k = probs.argmax()
            return (label, k)
        # 
        filename_prefix = 'kmeans-classification-results-%03d' % num_clusters
        #
        data = train_data.map(classify_sample)
        y_true_and_pred = data.collect()
        y_true = numpy.array([x[0] for x in y_true_and_pred])
        y_pred = numpy.array([x[1] for x in y_true_and_pred])
        save_results('results.digits.train', filename_prefix, y_true, y_pred)
        #
        data = test_data.map(classify_sample)
        y_true_and_pred = data.collect()
        y_true = numpy.array([x[0] for x in y_true_and_pred])
        y_pred = numpy.array([x[1] for x in y_true_and_pred])
        save_results('results.digits.test',  filename_prefix, y_true, y_pred)
    ####################################################################
    rdd_train.unpersist()
    sc.stop()
