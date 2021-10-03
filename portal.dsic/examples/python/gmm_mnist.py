"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://personales.upv.es/jon)
    Version: 1.0
    Date: October 2021
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

    Using E-M algorithm for Unsupervised Maximum Likelihood Estimation

"""

import os
import sys
import numpy

import machine_learning
from utils_for_results import save_results
from load_mnist import load_mnist
from sklearn.decomposition import PCA
from pyspark import SparkContext, SparkConf


if __name__ == "__main__":

    """
    Usage: spark-submit --master local[4]  python/gmm_mnist.py  --base-dir .   --dataset data/uc13-train.csv  --covar full      --max-components  50  2>/dev/null
           spark-submit --master local[4]  python/gmm_mnist.py  --base-dir .   --dataset data/uc13-train.csv  --covar diagonal  --max-components 300  2>/dev/null
    """

    verbose = 0
    covar_type = 'diagonal'
    max_components = 300
    base_dir = '.'
    standalone = False
    spark_context = None
    num_partitions = 80
    gmm_filename = None
    batch_size = 500
    do_compute_confusion_matrix = False
    do_classification = False
    results_dir = 'results.digits.train'
    models_dir = 'models.digits'
    log_dir = 'log.digits'
                                                   
    for i in range(len(sys.argv)):
        if sys.argv[i] == "--covar":
            covar_type = sys.argv[i + 1]
        elif sys.argv[i] == "--max-components":
            max_components = int(sys.argv[i + 1])
        elif sys.argv[i] == "--base-dir":
            base_dir = sys.argv[i + 1]
        elif sys.argv[i] == "--verbosity":
            verbose = int(sys.argv[i + 1])
        elif sys.argv[i] == "--standalone":
            standalone = True
        elif sys.argv[i] == "--num-partitions":
            num_partitions = int(sys.argv[i + 1])
        elif sys.argv[i] == "--batch-size":
            batch_size = int(sys.argv[i + 1])
        elif sys.argv[i] == "--model":
            gmm_filename = sys.argv[i + 1]
        elif sys.argv[i] == "--compute-confusion-matrix":
            do_compute_confusion_matrix = True
        elif sys.argv[i] == "--classify":
            do_classification = True
        elif sys.argv[i] == "--predict":
            do_prediction = True
        elif sys.argv[i] == "--results-dir":
            results_dir = sys.argv[i + 1]

    spark_conf = SparkConf().set("spark.driver.maxResultSize", "24g").set("spark.app.name", "GMM-MLE-dataset-MNIST")
    #spark_context = SparkContext(appName = "GMM-MLE-dataset-MNIST")
    spark_context = SparkContext(conf = spark_conf)

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

    os.makedirs(base_dir + '/' + log_dir,    exist_ok = True)
    os.makedirs(base_dir + '/' + models_dir, exist_ok = True)

    rdd_train = spark_context.parallelize([(y, x.copy()) for x, y in zip(X_train, y_train)], numSlices = num_partitions)
    rdd_test  = spark_context.parallelize([(y, x.copy()) for x, y in zip(X_test, y_test)], numSlices = num_partitions)
    num_samples = rdd_train.count()
    print(f'train subset with {num_samples} distributed into {rdd_train.getNumPartitions()} partitions')
    print(f'test  subset with {rdd_test.count()} distributed into {rdd_test.getNumPartitions()} partitions')

    if do_compute_confusion_matrix:
        assert gmm_filename is not None
        gmm = machine_learning.GMM()
        gmm.load_from_text(gmm_filename)

        def samples_to_probs(t):
            patient, label, x = t
            probs, logL = gmm.posteriors(x) # this works with one sample per column, so the transpose should be provided
            return (label, probs) # because the rows are the number of components in the GMM

        data = rdd_train.map(samples_to_probs)

        matrix = data.reduceByKey(lambda x, y: x + y).collect()

        accumulators = numpy.zeros([len(matrix), gmm.n_components])
        for row in matrix:
            l = row[0]
            x = row[1]
            accumulators[l] += x

        f = open(f'{models_dir}/gmm-distribution-%04d.csv' % gmm.n_components, 'wt')
        for l in range(len(accumulators)):
            print(";".join("{:f}".format(v) for v in accumulators[l]), file = f)
        f.close()
        
    elif do_classification:
        assert gmm_filename is not None
        gmm = machine_learning.GMM()
        gmm.load_from_text(gmm_filename)

        filename = f'{models_dir}/gmm-distribution-%04d.csv' % gmm.n_components
        accumulators = numpy.genfromtxt(filename, delimiter = ';')
        conditional_probabilities = accumulators / accumulators.sum(axis = 1).reshape(-1 ,1)
        target_classes_a_priori_probabilities = accumulators.sum(axis = 1) / accumulators.sum()

        def classify_sample(t):
            label, x = t
            _log_densities = gmm.log_densities(x, with_a_priori_probs = False) # J x 1
            _densities = numpy.exp(_log_densities - _log_densities.max()) # J x 1
            _probs = numpy.dot(conditional_probabilities, _densities).T # 1 x K
            _probs *= target_classes_a_priori_probabilities
            return (label, _probs.argmax())

        filename_prefix = 'gmm-classification-results-%04d' % gmm.n_components
        #
        data = rdd_train.map(classify_sample)
        y_true_and_pred = data.collect()
        y_true = numpy.array([x[0] for x in y_true_and_pred])
        y_pred = numpy.array([x[1] for x in y_true_and_pred])
        save_results('results.digits.train', filename_prefix, y_true, y_pred)
        #
        data = rdd_test.map(classify_sample)
        y_true_and_pred = data.collect()
        y_true = numpy.array([x[0] for x in y_true_and_pred])
        y_pred = numpy.array([x[1] for x in y_true_and_pred])
        save_results('results.digits.test', filename_prefix, y_true, y_pred)

    else:
        ####################################################################
        data = rdd_train.map(lambda x: x[1])

        K = (num_samples + batch_size - 1) / batch_size 
        K = ((K // num_partitions) + 1) * num_partitions
        #samples = text_lines.map(lambda line: (numpy.random.randint(K), numpy.array([float(x) for x in line.split()])))
        samples = data.map(lambda x: (numpy.random.randint(K), x))

        # Shows an example of each element in the temporary RDD of tuples [key, sample]
        if verbose > 1:
            x = data.first()
            print(x)
            print(type(x))

        """
            Thanks to the random integer number used as key we can build a new RDD of blocks
            of samples, where each block contains approximately the number of samples specified
            in batch_size.
        """
        samples = samples.reduceByKey(lambda x, y: numpy.vstack([x, y]))

        # Repartition if necessary
        if samples.getNumPartitions() < num_partitions:
            samples = samples.repartition(num_partitions)
            print("rdd repartitioned to %d partitions" % samples.getNumPartitions())

        # Shows an example of each element in the temporary RDD of tuples [key, block of samples]
        if verbose > 1:
            print(samples.first())
            print(type(samples.first()))

        """
            Convert the RDD of tuples to the definitive RDD of blocks of samples
        """
        samples = samples.map(lambda x: x[1])

        # Shows an example of each element in the temporary RDD of blocks of samples
        if verbose > 1:
            print(samples.first())
            print(type(samples.first()))

        samples.persist()
        print("we are working with %d blocks of approximately %d samples " % (samples.count(), batch_size))

        # Shows an example of shape of the elements in the temporary RDD of blocks of samples
        if verbose > 0:
            print(samples.first().shape)
        # Gets the dimensionality of samples in order to create the object of the class MLE.
        dim_x = samples.first().shape[1]

        mle = machine_learning.MLE(covar_type = covar_type, dim = dim_x, log_dir = base_dir + '/' + log_dir, models_dir = base_dir + '/' + models_dir)
        if gmm_filename is not None:
            mle.gmm.load_from_text(gmm_filename)

        mle.fit_with_spark(spark_context = spark_context, samples = samples, max_components = max_components)

        samples.unpersist()
    #
    spark_context.stop()
