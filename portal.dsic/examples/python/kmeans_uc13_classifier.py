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

from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix

from machine_learning import KMeans

from matplotlib import pyplot

try:
    from pyspark import SparkContext
except:
    pass

class MyArgmaxForPredictedLabels(BaseEstimator):
    def __init__(self, threshold = 0.5):
        self.threshold = threshold
        self._estimator_type = 'classifier'

    def fit(self, X, y):
        raise Exception('No fit implemented in this class')
        return self

    def predict(self, y_probs):
        assert type(y_probs) == numpy.ndarray
        assert len(y_probs.shape) == 1
        return y_probs



if __name__ == "__main__":

    """
    Usage: spark-submit --master local[4]  python/kmeans_uc13_classifier.py  \
                                                                --dataset data/uc13.csv \
                                                                --codebook models/kmeans_model-uc13-1000.pkl \
                                                                --confusion-matrix models/cluster-distribution-1000.csv 2>/dev/null
    """
    label_mapping = [i for i in range(10)]

    verbose = 0
    dataset_filename = 'data/uc13-train.csv'
    codebook_filename = 'models/kmeans_model-uc13-200.pkl'
    counter_pairs_filename = 'models/cluster-distribution-200.csv'
    spark_context = None
    num_partitions = 40
    batch_size = 100
    num_channels = 21
    results_dir = 'results3.train'
    models_dir = 'models'
    log_dir = 'log'
    do_reshape = True
    do_standard_scaling = True
                                                   
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
        elif sys.argv[i] == "--counter-pairs":
            counter_pairs_filename = sys.argv[i + 1]
        elif sys.argv[i] == "--results-dir":
            results_dir = sys.argv[i + 1]
        elif sys.argv[i] == "--reduce-labels":
            label_mapping[3] = 0
            label_mapping[4] = 0
            label_mapping[5] = 0
            label_mapping[6] = 1
            label_mapping[7] = 0
            label_mapping[8] = 0
            label_mapping[9] = 0
        elif sys.argv[i] == "--from-pca":
            models_dir = 'models.pca'
            log_dir = 'log.pca'
            do_reshape = False
            do_standard_scaling = False

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

    # Load and parse the data
    csv_lines = spark_context.textFile(dataset_filename)
    print("file(s) loaded ")
    if csv_lines.getNumPartitions() < num_partitions:
        csv_lines = csv_lines.repartition(num_partitions)
    #csv_lines.persist()
    num_samples = csv_lines.count()
    print("loaded %d samples distributed in %d partitions" % (num_samples, csv_lines.getNumPartitions()))
    #csv_lines.unpersist()


    def csv_line_to_patient_label_and_sample_reshape(line):
        parts = line.split(';')
        return (parts[0], label_mapping[int(parts[1])], numpy.array([float(x) for x in parts[2:]]).reshape(num_channels, 14))

    def csv_line_to_patient_label_and_sample(line):
        parts = line.split(';')
        return (parts[0], label_mapping[int(parts[1])], numpy.array([float(x) for x in parts[2:]]))

    if do_reshape:
        data = csv_lines.map(csv_line_to_patient_label_and_sample_reshape)
    else:
        data = csv_lines.map(csv_line_to_patient_label_and_sample)

    '''
    x = data.first()
    print(x[0])
    print(x[1])
    print(x[2].shape)
    '''

    if do_standard_scaling:
        ####################################################################
        # begin: do standard scaling 
        ####################################################################
        statistics_filename = 'models/mean_and_std.pkl'
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

    def classify_sample(t):
        patient, label, sample = t
        if do_reshape:
            cluster_assignment = kmeans.predict(sample)
        else:
            cluster_assignment = kmeans.predict([sample])
        probs = numpy.zeros(conditional_probabilities.shape[0]) # one per target class
        for j in cluster_assignment:
            probs += conditional_probabilities[:, j] 
        #probs *= target_class_a_priori_probabilities
        k = probs.argmax()
        return (patient, label, k)
        
    data = data.map(classify_sample)

    y_true_and_pred = data.collect()
    y_true = numpy.array([x[1] for x in y_true_and_pred])
    y_pred = numpy.array([x[2] for x in y_true_and_pred])


    os.makedirs(results_dir, exist_ok = True)
    f_results = open(f'{results_dir}/classification-results-%03d.txt' % kmeans.n_clusters, 'wt')
    _cm_ = confusion_matrix(y_true, y_pred)
    for i in range(_cm_.shape[0]):
        for j in range(_cm_.shape[1]):
            f_results.write(' %10d' % _cm_[i, j])
        f_results.write('\n')
    f_results.write('\n')
    print(classification_report(y_true, y_pred), file = f_results)
    f_results.close()
    #
    fig, axes = pyplot.subplots(nrows = 1, ncols = 2, figsize = (16, 7))
    #fig.suptitle(title)
    #
    plot_confusion_matrix(estimator = MyArgmaxForPredictedLabels(),
                          X = y_pred, y_true = y_true,
                          normalize = 'true', ax = axes[0], cmap = 'Blues', colorbar = False)
    #
    plot_confusion_matrix(estimator = MyArgmaxForPredictedLabels(),
                          X = y_pred, y_true = y_true,
                          normalize = 'pred', ax = axes[1], cmap = 'Oranges', colorbar = False)
    #
    pyplot.tight_layout()
    pyplot.savefig(f'{results_dir}/classification-results-%03d.svg' % kmeans.n_clusters, format = 'svg')
    pyplot.savefig(f'{results_dir}/classification-results-%03d.png' % kmeans.n_clusters, format = 'png')
    del fig

    spark_context.stop()
