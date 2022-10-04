"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://personales.upv.es/jon)
    Version: 2.0
    Date: October 2022
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

    Using E-M algorithm for Unsupervised Maximum Likelihood Estimation

"""

import os
import sys
import numpy
import pickle
import tempfile

from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
from matplotlib import pyplot
from utils_for_results import save_results

import machine_learning

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


def load_samples_from_file(filename):
    X = list()
    Y = list()
    with open(filename, "rt") as f:
        values = [float(x) for x in line.split(sep = ';')]
        X.append(values[1:])
        Y.append(int(values[0])) # label in this dataset is at position 0
    X = numpy.array(X)
    Y = numpy.array(Y)
    return X, Y

def load_samples(index_filename):
    f = open(index_filename, "rt")
    X = []
    Y = []
    for filename in f:
        print("loading file " + filename.strip())
        _x_, _y_ = load_samples_from_file(filename.strip())
        X.append(_x_)
        Y.append(_y_)
        #print( _x_.shape )
    f.close()
    return numpy.vstack(X), numpy.vstack(Y)


def estimate_gmm(data, spark_context, max_components, models_dir = None, log_dir = None, batch_size = 500):
    num_partitions = data.getNumPartitions()
    K = (data.count() + batch_size - 1) / batch_size 
    K = ((K // num_partitions) + 1) * num_partitions
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

    temp_dirname = tempfile.mkdtemp(prefix = 'mle', suffix = '.gmm')

    mle = machine_learning.MLE( covar_type = covar_type,
                                dim = dim_x,
                                log_dir = log_dir if log_dir is not None else temp_dirname,
                                models_dir = models_dir if models_dir is not None else temp_dirname)
    mle.fit_with_spark( spark_context = spark_context,
                        samples = samples,
                        max_components = max_components,
                        epsilon = 1.0e-4)
    samples.unpersist()
    #
    return mle.gmm
# ------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    """
    Usage: spark-submit --master local[4]  python/gmm_uc13_21x20.py  --base-dir .   --patient chb01  --covar full      --max-components  10 2>/dev/null
           spark-submit --master local[4]  python/gmm_uc13_21x20.py  --base-dir .   --patient chb03  --covar diagonal  --max-components  30 2>/dev/null
    """

    label_mapping = [i for i in range(10)]

    verbose = 0
    base_dir = 'uc13-21x20'
    patient = 'chb01'
    covar_type = 'diagonal'
    max_components = 30
    standalone = False
    spark_context = None
    num_partitions = 60
    gmm_filename = None
    num_channels = 21
    batch_size = 24 * num_channels
    #
    do_binary_classification = False
    do_classification = False
    do_prediction = False

    for i in range(len(sys.argv)):
        if sys.argv[i] == "--patient": patient = sys.argv[i + 1]
        elif sys.argv[i] == "--covar": covar_type = sys.argv[i + 1]
        elif sys.argv[i] == "--max-components": max_components = int(sys.argv[i + 1])
        elif sys.argv[i] == "--base-dir": base_dir = sys.argv[i + 1]
        elif sys.argv[i] == "--verbosity": verbose = int(sys.argv[i + 1])
        elif sys.argv[i] == "--standalone": standalone = True
        elif sys.argv[i] == "--num-partitions": num_partitions = int(sys.argv[i + 1])
        elif sys.argv[i] == "--batch-size": batch_size = int(sys.argv[i + 1])
        elif sys.argv[i] == "--model": gmm_filename = sys.argv[i + 1]
        elif sys.argv[i] == "--classify": do_classification = True
        elif sys.argv[i] == "--predict": do_prediction = True
        elif sys.argv[i] == "--do-binary-classification": do_binary_classification = True



    if do_binary_classification:
        label_mapping[2] = 0
        label_mapping[3] = 0
        label_mapping[4] = 0
        label_mapping[5] = 0
        label_mapping[6] = 0
        label_mapping[7] = 0
        label_mapping[8] = 0
        label_mapping[9] = 0


    if not standalone and not do_prediction:
        spark_context = SparkContext(appName = "GMM-MLE-dataset-UC13")

    def csv_line_to_patient_label_and_sample(line):
        parts = line.split(';')
        return (parts[0], label_mapping[int(parts[1])], numpy.array([float(x) for x in parts[2:]]))


    dataset_filename_train = f'data/uc13/uc13-{patient}-21x20-train-pca.csv'
    dataset_filename_test  = f'data/uc13/uc13-{patient}-21x20-test-pca.csv'
    labels = numpy.unique(label_mapping)

    models_dir  = f'{base_dir}/{patient}/models'
    log_dir     = f'{base_dir}/{patient}/log'
    results_dir = f'{base_dir}/{patient}/results'

    if do_binary_classification:
        results_dir += '.02-classes'
    else:
        results_dir += '.10-classes'

    os.makedirs(models_dir, exist_ok = True)
    os.makedirs(log_dir,    exist_ok = True)

    if spark_context is not None:

        """
            Load all the lines in a file (or files in a directory) into an RDD of text lines.

            It is assumed there is no header, each CSV file contains an undefined number or lines.
            - Each line represents a sample.
            - All the lines **must** contain the same number of values.
            - All the values **must** be numeric, integers or real values.
        """
        # Load and parse the data
        rdd_train = spark_context.textFile(dataset_filename_train).map(csv_line_to_patient_label_and_sample)
        rdd_test  = spark_context.textFile(dataset_filename_test).map(csv_line_to_patient_label_and_sample)
        data_rdds = dict()
        data_rdds['train'] = rdd_train
        data_rdds['test'] = rdd_test
        print("file(s) loaded ")
        #csv_lines = csv_lines.repartition(num_partitions)

        """
            Convert the text lines into numpy arrays.

            Taking as input the RDD text_lines, a map operation is applied to each text line in order
            to convert it into a numpy array, as a result a new RDD of numpy arrays is obtained.
            
            Nevertheless, as we need an RDD with blocks of samples instead of single samples, we 
            associate with each sample a random integer number in a specific range.

            So, instead of an RDD with of numpy arrays we get an RDD with tuples [ int, numpy.array ]
        """

        if do_binary_classification:
            models_filename = f'{models_dir}/gmm-02-target-classes.pkl'
        else:
            models_filename = f'{models_dir}/gmm-10-target-classes.pkl'
        # do GMM MLE for each target class

        if os.path.exists(models_filename):
            with open(models_filename, 'rb') as f:
                model = pickle.load(f)
                f.close()
        else:
            model = dict()
            for label in labels:
                data = rdd_train.filter(lambda x: x[1] == label).map(lambda x: x[2])
                print(f'working with {data.count()} samples from target class {label} distributed in {data.getNumPartitions()} partitions')
                gmm = None
                if data.count() > 1000:
                    gmm = estimate_gmm(data = data, spark_context = spark_context, max_components = max_components)
                elif data.count() > 100:
                    gmm = estimate_gmm(data = data, spark_context = spark_context, max_components = 2)
                elif data.count() > 0:
                    gmm = estimate_gmm(data = data, spark_context = spark_context, max_components = 1)
                #
                model[label] = gmm
            #
            with open(models_filename, 'wb') as f:
                pickle.dump(model, f)
                f.close()
            

        #target_classes_a_priori_probabilities = accumulators.sum(axis = 1) / accumulators.sum()

        def classify_sample(t):
            patient, label, x = t
            _probs = list()
            for i in labels:
                gmm = model[i]
                if gmm is not None:
                    _log_densities = gmm.log_densities(x, with_a_priori_probs = True)
                    _densities = numpy.exp(_log_densities - _log_densities.max()) * numpy.exp(_log_densities.max())
                    _probs.append(_densities.sum())
                else:
                    _probs.append(0.0)
            # for
            _probs = numpy.array(_probs)
            return (patient, label, _probs.argmax())

        for subset in ['train', 'test']:
            y_true_and_pred = data_rdds[subset].map(classify_sample).collect()
            y_true = numpy.array([x[1] for x in y_true_and_pred])
            y_pred = numpy.array([x[2] for x in y_true_and_pred])

            filename_prefix = f'gmm-{patient}-21x20-%04d' % max_components
            save_results(f'{results_dir}.{subset}', filename_prefix, y_true, y_pred, labels = labels)

        spark_context.stop()

'''
    elif do_prediction:
        assert gmm_filename is not None
        gmm = machine_learning.GMM()
        gmm.load_from_text(gmm_filename)

        filename = f'{models_dir}/gmm-distribution-%04d.csv' % gmm.n_components
        accumulators = numpy.genfromtxt(filename, delimiter = ';')
        conditional_probabilities = accumulators.copy()
        for i in range(conditional_probabilities.shape[0]):
            conditional_probabilities[i, :] /= conditional_probabilities[i, :].sum()

        target_classes_a_priori_probabilities = accumulators.sum(axis = 1) / accumulators.sum()

        def classify_sample(t):
            patient, label, x = t
            if len(x.shape) == 2:
                _log_densities = gmm.log_densities_batch(x.T, with_a_priori_probs = False) # J x N
                _densities = numpy.exp(_log_densities - _log_densities.max(axis = 0)) # J x N
                _probs = numpy.dot(conditional_probabilities, _densities).T # N x K
                _probs = _probs.sum(axis = 0) # * target_classes_a_priori_probabilities
            else:
                _log_densities = gmm.log_densities(x, with_a_priori_probs = False) # J x 1
                _densities = numpy.exp(_log_densities - _log_densities.max()) # J x 1
                _probs = numpy.dot(conditional_probabilities, _densities).T # 1 x K
            #_probs *= target_classes_a_priori_probabilities

            return (patient, label, _probs)

        y_true_and_pred = list()
        #
        sliding_window_length = 5 * 60 # 1800 seconds -> 30 minutes
        sliding_window_step = 60 * 5 # 300 seconds -> 5 minute
        #
        target_class_accumulators = numpy.zeros(conditional_probabilities.shape[0])
        list_of_predictions = list()
        list_of_alarms = list()
        current_time = 0
        state = 'inter-ictal'
        
        old_patient = "non-existent-yet"
        for line in sys.stdin:
            if do_reshape:
                patient, label, label_probs = classify_sample(csv_line_to_patient_label_and_sample_reshape(line))
            else:
                patient, label, label_probs = classify_sample(csv_line_to_patient_label_and_sample(line))
            #
            if patient != old_patient:
                # reset variables
                # 
                old_patient = patient
                target_class_accumulators = numpy.zeros(conditional_probabilities.shape[0])
                list_of_predictions = list()
                list_of_alarms = list()
                current_time = 0
                state = 'inter-ictal'
                for pred_label, pred_time in list_of_alarms:
                    y_true_and_pred.append((0, pred_label))
            #
            list_of_predictions.append(label_probs)
            target_class_accumulators += label_probs
            if len(list_of_predictions) > sliding_window_length:
                target_class_accumulators -= list_of_predictions[0]
                del list_of_predictions[0]
            # 
            target_class_probs = target_class_accumulators / sum(target_class_accumulators)
            if current_time > 0 and current_time % sliding_window_step == 0:
                print(patient, label, " ".join("{:8.2f}".format(100 * x) for x in target_class_probs))
            #
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
                    #if lower_threshold_for_target_class_2 <= target_class_probs[2] and \
                    #   lower_threshold_for_target_class_1 <= target_class_probs[1] <= upper_threshold_for_target_class_1:
                    if target_class_probs[6:].sum() > 0.40 or \
                       target_class_probs[1:5].sum() > 0.40:
                        list_of_alarms.append((1, current_time))
                    else:
                        list_of_alarms.append((0, current_time))
                    #
                # 
            current_time += 2 # add 2 seconds because each sample comes 2 seconds after the previous one
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
        f_results = open(f'{results_dir}/gmm-prediction-results-%04d.txt' % gmm.n_components, 'wt')
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
        pyplot.savefig(f'{results_dir}/gmm-prediction-results-%04d.svg' % gmm.n_components, format = 'svg')
        pyplot.savefig(f'{results_dir}/gmm-prediction-results-%04d.png' % gmm.n_components, format = 'png')
        del fig
'''
