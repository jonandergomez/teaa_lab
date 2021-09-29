"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://personales.upv.es/jon)
    Version: 1.0
    Date: September 2021
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

    Using E-M algorithm for Unsupervised Maximum Likelihood Estimation

"""

import os
import sys
import numpy

from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
from matplotlib import pyplot

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

if __name__ == "__main__":

    """
    Usage: spark-submit --master local[4]  python/gmm_uc13.py  --base-dir .   --dataset data/uc13-train.csv  --covar full      --max-components  50  2>/dev/null
           spark-submit --master local[4]  python/gmm_uc13.py  --base-dir .   --dataset data/uc13-train.csv  --covar diagonal  --max-components 300  2>/dev/null
    """

    label_mapping = [i for i in range(10)]

    verbose = 0
    covar_type = 'diagonal'
    max_components = 300
    dataset_filename = 'data/uc13-train.csv'
    base_dir = '.'
    standalone = False
    spark_context = None
    num_partitions = 80
    gmm_filename = None
    num_channels = 21
    batch_size = 24 * num_channels
    do_compute_confusion_matrix = False
    do_classification = False
    do_prediction = False
    results_dir = 'results3.train'
    models_dir = 'models'
    log_dir = 'log'
    do_reshape = True
                                                   
    for i in range(len(sys.argv)):
        if sys.argv[i] == "--covar":
            covar_type = sys.argv[i + 1]
        elif sys.argv[i] == "--max-components":
            max_components = int(sys.argv[i + 1])
        elif sys.argv[i] == "--base-dir":
            base_dir = sys.argv[i + 1]
        elif sys.argv[i] == "--dataset":
            dataset_filename = sys.argv[i + 1]
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

    if not standalone and not do_prediction:
        spark_context = SparkContext(appName = "GMM-MLE-dataset-UC13")


    os.makedirs(base_dir + '/' + log_dir,    exist_ok = True)
    os.makedirs(base_dir + '/' + models_dir, exist_ok = True)

    if spark_context is not None:

        """
            Load all the lines in a file (or files in a directory) into an RDD of text lines.

            It is assumed there is no header, each CSV file contains an undefined number or lines.
            - Each line represents a sample.
            - All the lines **must** contain the same number of values.
            - All the values **must** be numeric, integers or real values.
        """
        # Load and parse the data
        csv_lines = spark_context.textFile(dataset_filename)
        print("file(s) loaded ")
        csv_lines = csv_lines.repartition(num_partitions)

        """
            Convert the text lines into numpy arrays.

            Taking as input the RDD text_lines, a map operation is applied to each text line in order
            to convert it into a numpy array, as a result a new RDD of numpy arrays is obtained.
            
            Nevertheless, as we need an RDD with blocks of samples instead of single samples, we 
            associate with each sample a random integer number in a specific range.

            So, instead of an RDD with of numpy arrays we get an RDD with tuples [ int, numpy.array ]
        """

        def csv_line_to_patient_label_and_sample_reshape(line):
            parts = line.split(';')
            return (parts[0], label_mapping[int(parts[1])], numpy.array([float(x) for x in parts[2:]]).reshape(num_channels, 14))

        def csv_line_to_patient_label_and_sample(line):
            parts = line.split(';')
            return (parts[0], label_mapping[int(parts[1])], numpy.array([float(x) for x in parts[2:]]))

        if do_reshape:
            data = csv_lines.map(csv_line_to_patient_label_and_sample_reshape)
        else
            data = csv_lines.map(csv_line_to_patient_label_and_sample)
        num_samples = data.count()
        x = data.take(1)
        dim = x[0][2].shape[1]

        print(f'loaded {num_samples} {dim}-dimensional samples into {data.getNumPartitions()} partitions')

        if do_compute_confusion_matrix:
            assert gmm_filename is not None
            gmm = machine_learning.GMM()
            gmm.load_from_text(gmm_filename)

            def samples_to_probs(t):
                patient, label, x = t
                probs, logL = gmm.posteriors_batch(x.T) # this works with one sample per column, so the transpose should be provided
                return (label, probs.sum(axis = 1)) # because the rows are the number of components in the GMM

            data = data.map(samples_to_probs)

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
            conditional_probabilities = accumulators.copy()
            for i in range(conditional_probabilities.shape[0]):
                conditional_probabilities[i, :] /= conditional_probabilities[i, :].sum()

            target_classes_a_priori_probabilities = accumulators.sum(axis = 1) / accumulators.sum()

            def classify_sample(t):
                patient, label, x = t
                _log_densities = gmm.log_densities_batch(x.T, with_a_priori_probs = False) # J x N
                _densities = numpy.exp(_log_densities - _log_densities.max(axis = 0)) # J x N
                _probs = numpy.dot(conditional_probabilities, _densities).T # N x K
                _probs = _probs.sum(axis = 0) # * target_classes_a_priori_probabilities

                return (patient, label, _probs.argmax())

            data = data.map(classify_sample)

            y_true_and_pred = data.collect()
            y_true = numpy.array([x[1] for x in y_true_and_pred])
            y_pred = numpy.array([x[2] for x in y_true_and_pred])


            os.makedirs(results_dir, exist_ok = True)
            f_results = open(f'{results_dir}/gmm-classification-results-%04d.txt' % gmm.n_components, 'wt')
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
            pyplot.savefig(f'{results_dir}/gmm-classification-results-%04d.svg' % gmm.n_components, format = 'svg')
            pyplot.savefig(f'{results_dir}/gmm-classification-results-%04d.png' % gmm.n_components, format = 'png')
            del fig

        else:
            ####################################################################
            data = data.map(lambda x: x[2])

            batch_size = max(1, batch_size // num_channels)

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

        def csv_line_to_patient_label_and_sample(line):
            parts = line.split(';')
            return (parts[0], label_mapping[int(parts[1])], numpy.array([float(x) for x in parts[2:]]).reshape(num_channels, 14))

        def classify_sample(t):
            patient, label, x = t
            _log_densities = gmm.log_densities_batch(x.T, with_a_priori_probs = False) # J x N
            _densities = numpy.exp(_log_densities - _log_densities.max(axis = 0)) # J x N
            _probs = numpy.dot(conditional_probabilities, _densities).T # N x K
            _probs = _probs.sum(axis = 0) #* target_classes_a_priori_probabilities
            _probs /= _probs.sum()

            #return (patient, label, _probs.argmax())
            return (patient, label, _probs)

        y_true_and_pred = list()
        #
        sliding_window_length = 30 * 60 # 1800 seconds -> 30 minutes
        sliding_window_step = 60 * 5 # 300 seconds -> 5 minute
        lower_threshold_for_target_class_2 = 0.10
        lower_threshold_for_target_class_1 = 0.05
        upper_threshold_for_target_class_1 = 0.95
        #
        target_class_accumulators = numpy.zeros(conditional_probabilities.shape[0])
        list_of_predictions = list()
        list_of_alarms = list()
        current_time = 0
        state = 'inter-ictal'
        
        old_patient = "non-existent-yet"
        for line in sys.stdin:
            #patient, label, predicted_label = classify_sample(csv_line_to_patient_label_and_sample(line))
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
                       target_class_probs[1:4].sum() > 0.30:
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
        f_results = open(f'{results_dir}/gmm-prediction-results-%03d.txt' % gmm.n_components, 'wt')
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
        pyplot.savefig(f'{results_dir}/gmm-prediction-results-%03d.svg' % gmm.n_components, format = 'svg')
        pyplot.savefig(f'{results_dir}/gmm-prediction-results-%03d.png' % gmm.n_components, format = 'png')
        del fig
