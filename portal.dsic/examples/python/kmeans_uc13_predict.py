"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://personales.upv.es/jon)
    Version: 1.0
    Date: September 2021
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

    K-Means-based sequence-based predictor

"""

import os
import sys
import numpy
import pickle

from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix

from machine_learning import KMeans

from matplotlib import pyplot


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
    run with this command:

        hdfs dfs -cat  data/uc13-train.csv | python python/kmeans_uc13_predict.py \
                                                                --codebook models/kmeans_model-uc13-1000.pkl \
                                                                --counter-pairs models/cluster-distribution-1000.csv 2>/dev/null
    """
    label_mapping = [i for i in range(10)]

    verbose = 0
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
        if sys.argv[i] == "--verbosity":
            verbose = int(sys.argv[i + 1])
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

    def csv_line_to_patient_label_and_sample_reshape(line):
        parts = line.split(';')
        return (parts[0], label_mapping[int(parts[1])], numpy.array([float(x) for x in parts[2:]]).reshape(num_channels, 14))

    def csv_line_to_patient_label_and_sample(line):
        parts = line.split(';')
        return (parts[0], label_mapping[int(parts[1])], numpy.array([float(x) for x in parts[2:]]))

    ####################################################################
    # begin: load standard scaling 
    ####################################################################
    statistics_filename = f'{models_dir}/mean_and_std.pkl'
    if do_standard_scaling and os.path.exists(statistics_filename):
        with open(statistics_filename, 'rb') as f:
            stats = pickle.load(f)
            f.close()
        x_mean, x_std = stats
    else:
        x_mean = numpy.zeros(14)
        x_std = numpy.ones(14)
    #
    ####################################################################
    # end: load standard scaling 
    ####################################################################

    def classify_sample(t):
        patient, label, sample = t
        if do_standard_scaling:
            sample = (sample - x_mean) / x_std

        if do_reshape:
            cluster_assignment = kmeans.predict(sample)
            probs = numpy.zeros(conditional_probabilities.shape[0]) # one per target class
            for j in cluster_assignment:
                probs += conditional_probabilities[:, j] 
        else:
            cluster_assignment = kmeans.predict([sample])
            j = cluster_assignment[0]
            probs = conditional_probabilities[:, j] 
        #probs *= target_class_a_priori_probabilities
        k = probs.argmax()
        return (patient, label, k)
        

    y_true_and_pred = list()
    #
    sliding_window_length = 5 * 60 # 1800 seconds -> 30 minutes
    sliding_window_step = 60 * 5 # 300 seconds -> 5 minute
    #
    target_class_counters = [0] * conditional_probabilities.shape[0]
    list_of_predictions = list()
    list_of_alarms = list()
    current_time = 0
    state = 'inter-ictal'
    
    old_patient = "non-existent-yet"
    for line in sys.stdin:
        if do_reshape:
            patient, label, predicted_label = classify_sample(csv_line_to_patient_label_and_sample_reshape(line))
        else:
            patient, label, predicted_label = classify_sample(csv_line_to_patient_label_and_sample(line))
        #
        if patient != old_patient:
            # reset variables
            # 
            old_patient = patient
            target_class_counters = [0] * conditional_probabilities.shape[0]
            list_of_predictions = list()
            list_of_alarms = list()
            current_time = 0
            state = 'inter-ictal'
        #
        list_of_predictions.append(predicted_label)
        target_class_counters[predicted_label] += 1
        if len(list_of_predictions) > sliding_window_length:
            target_class_counters[list_of_predictions[0]] -= 1
            del list_of_predictions[0]
        # 
        target_class_probs = numpy.array([float(x) for x in target_class_counters]) / sum(target_class_counters)
        #
        current_time += 2 # add 2 seconds because each sample comes 2 seconds after the previous one
        #
        if current_time > 0 and current_time % sliding_window_step == 0:
            print(patient, label, " ".join("{:8.2f}".format(100 * x) for x in target_class_probs))

        if label == 1: # an ictal period is reached
            if state == 'inter-ictal':
                print('ictal period starts, press enter ...')
                for pred_label, pred_time in list_of_alarms:
                    if current_time - pred_time <= 60 * 60: # 3600 seconds -> one hour
                        y_true_and_pred.append((1, pred_label))
                    else:
                        y_true_and_pred.append((0, pred_label))
                #
                state = 'ictal'
                list_of_alarms = list()
            #
        elif label in [0, 2, 3, 4, 5]: # exclude post-ictal periods
            state = 'inter-ictal'
            #
            if current_time > 0 and current_time % sliding_window_step == 0:
                if target_class_probs[1:4].sum() > 0.33:
                    list_of_alarms.append((1, current_time))
                else:
                    list_of_alarms.append((0, current_time))
            # 
    #        
    #
    for pred_label, pred_time in list_of_alarms:
        y_true_and_pred.append((0, pred_label))
    #
    y_true = numpy.array([x[0] for x in y_true_and_pred])
    y_pred = numpy.array([x[1] for x in y_true_and_pred])


    #####################################################################
    ### presentation of results
    #####################################################################
    os.makedirs(results_dir, exist_ok = True)
    f_results = open(f'{results_dir}/prediction-results-%03d.txt' % kmeans.n_clusters, 'wt')
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
    pyplot.savefig(f'{results_dir}/prediction-results-%03d.svg' % kmeans.n_clusters, format = 'svg')
    pyplot.savefig(f'{results_dir}/prediction-results-%03d.png' % kmeans.n_clusters, format = 'png')
    del fig
