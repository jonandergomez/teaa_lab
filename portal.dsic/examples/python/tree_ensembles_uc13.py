"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://personales.upv.es/jon)
    Version: 1.0
    Date: October 2021
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

    Using ensembles based on decision trees for classification or regression
    Ensemble types:
        Random Forest
        Gradient Boosted Trees
        Extremely Randomized Trees
"""

import os
import sys
import numpy
import pickle

from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
from matplotlib import pyplot

from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.regression import LabeledPoint

from pyspark import SparkContext

# ---------------------------------------------------------------------------------------------------
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
# ---------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------
def save_results(results_dir, filename_prefix, y_true, y_pred):
    os.makedirs(results_dir, exist_ok = True)
    f_results = open(f'{results_dir}/{filename_prefix}.txt', 'wt')
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
    pyplot.savefig(f'{results_dir}/{filename_prefix}.svg', format = 'svg')
    pyplot.savefig(f'{results_dir}/{filename_prefix}.png', format = 'png')
    del fig
# ---------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Usage: spark-submit --master local[4]  python/tree_ensembles_uc13.py --dataset data/uc13-train.csv --num-trees <nt> --max-depth <md> --impurity <imp> 
    """

    impurity = 'gini'
    max_depth = 10
    max_bins = 32
    num_trees = 100 # for Random Forest
    num_iterations = 100 # for Gradient Boosting
    
    ensemble_type = 'random-forest'

    verbose = 0
    dataset_filename = 'data/uc13-train.csv'
    model_filename = None

    spark_context = None
    num_partitions = 80
    num_channels = 21
    results_dir = 'results.ensembles.train.1'
    models_dir = 'models.ensembles'
    log_dir = 'log.ensembles'
    do_training = False
    do_classification = False
    do_prediction = False
    learning_rate = 0.1
    label_mapping = [i for i in range(10)]
                                                   
    for i in range(len(sys.argv)):
        if   sys.argv[i] == "--dataset"       :  dataset_filename = sys.argv[i + 1]
        elif sys.argv[i] == "--verbosity"     :           verbose = int(sys.argv[i + 1])
        elif sys.argv[i] == "--num-partitions":    num_partitions = int(sys.argv[i + 1])
        elif sys.argv[i] == "--ensemble-type" :     ensemble_type = sys.argv[i + 1]
        elif sys.argv[i] == "--num-trees"     :         num_trees = int(sys.argv[i + 1])
        elif sys.argv[i] == "--num-iterations":    num_iterations = int(sys.argv[i + 1])
        elif sys.argv[i] == "--impurity"      :          impurity = sys.argv[i + 1]
        elif sys.argv[i] == "--max-depth"     :         max_depth = int(sys.argv[i + 1])
        elif sys.argv[i] == "--max-bins"      :          max_bins = int(sys.argv[i + 1])
        elif sys.argv[i] == "--model"         :    model_filename = sys.argv[i + 1]
        elif sys.argv[i] == "--learning-rate" :     learning_rate = float(sys.argv[i + 1])
        elif sys.argv[i] == "--train"         :       do_training = True
        elif sys.argv[i] == "--classify"      : do_classification = True
        elif sys.argv[i] == "--predict"       :     do_prediction = True
        elif sys.argv[i] == "--results-dir"   :       results_dir = sys.argv[i + 1]
        elif sys.argv[i] == "--models-dir"    :        models_dir = sys.argv[i + 1]
        elif sys.argv[i] == "--log-dir"       :           log_dir = sys.argv[i + 1]
        elif sys.argv[i] == "--reduce-labels":
            label_mapping[0] = 0
            label_mapping[1] = 1
            label_mapping[2] = 1
            label_mapping[3] = 0
            label_mapping[4] = 0
            label_mapping[5] = 0
            label_mapping[6] = 1
            label_mapping[7] = 0
            label_mapping[8] = 0
            label_mapping[9] = 0

    if model_filename is None:
        model_filename = f'{models_dir}/{ensemble_type}-{num_trees}-{impurity}-{max_depth}'

    spark_context = SparkContext(appName = "RandomForest-dataset-UC13")


    '''
        In order to use the Spark implementation of classifiers each data point must be
        an object of the class LabeledPoint.
        Basically, each object of this class has two attributes: label and features.
        For classification tasks, label should be an integer, for regression tasks label
        should be a floating-point number.
    '''
    def csv_line_to_labeled_point(line):
        parts = line.split(';')
        return LabeledPoint(label_mapping[int(parts[1])], numpy.array([float(x) for x in parts[2:]]))

    #os.makedirs(log_dir,    exist_ok = True)
    #os.makedirs(models_dir, exist_ok = True)

    '''
        Load all the lines from a file (or files in a directory) into an RDD of text lines.

        It is assumed there is no header, each CSV file contains an undefined number or lines.
        - Each line represents a sample.
        - All the lines **must** contain the same number of values.
        - All the values **must** be numeric, i.e., integers or real values.
    '''
    # Data loading and parsing
    csv_lines = spark_context.textFile(dataset_filename)
    print("file(s) loaded ")
    # RDD repartitioning
    csv_lines = csv_lines.repartition(num_partitions)

    '''
        Convert the text lines into objects of the class LabeledPoint, where features are numpy arrays.

        Taking as input the RDD text_lines, a map operation is applied to each text line in order
        to convert it into an object of the class LabeledPoint numpy array, as a result a new RDD
        of LabeledPoint objects is obtained.
    '''
    # RDD object conversion
    data = csv_lines.map(csv_line_to_labeled_point)
    # Retrives basic info from the RDD object
    num_samples = data.count()
    x = data.take(1)
    dim = x[0].features.shape[0]

    print(f'loaded {num_samples} {dim}-dimensional samples into {data.getNumPartitions()} partitions')


    if do_training:
        if ensemble_type == 'random-forest':
            model = RandomForest.trainClassifier(data,
                                                numClasses = len(numpy.unique(label_mapping)),
                                                categoricalFeaturesInfo = {}, # nothing to use here
                                                numTrees = num_trees,
                                                featureSubsetStrategy = 'auto',
                                                impurity = impurity,
                                                maxDepth = max_depth,
                                                maxBins = max_bins)
            model.save(spark_context, model_filename)

        elif ensemble_type == 'gradient-boosted-trees':
            model = GradientBoostedTrees.trainClassifier(data,
                                                        categoricalFeaturesInfo = {}, # nothing to use here
                                                        loss = 'logLoss',
                                                        learningRate = learning_rate,
                                                        numIterations = num_iterations,
                                                        maxDepth = max_depth,
                                                        maxBins = max_bins)
            model.save(spark_context, model_filename)

        elif ensemble_type == 'extra-trees':
            # Retrieves all the dataset in the RDD to a local list
            local_data = data.collect()
            X = numpy.array([item.features for item in local_data]) # get features
            y = numpy.array([item.label    for item in local_data]) # get labels
            del local_data # free memory, just in case
            #
            model = ExtraTreesClassifier(n_estimators = num_trees, criterion = impurity, max_depth = max_depth, n_jobs = -1, verbose = 1)
            model.fit(X, y)
            # Saves the model to local disk (in the master) using Pickle serialization for Python
            os.makedirs(models_dir, exist_ok = True)
            with open(model_filename, 'wb') as f:
                pickle.dump(model, f)
                f.close()
        else:
            raise Exception(f'{ensemble_type} is no a valid ensemble type')

            
    if do_classification:
        if ensemble_type == 'random-forest':
            # Loads the model
            model = RandomForestModel.load(spark_context, model_filename)
            # Classifies the samples
            y_pred = model.predict(data.map(lambda x: x.features))
            # Merge ground-truth with predictions
            y_true_and_pred = data.map(lambda x: x.label).zip(y_pred).collect()
            y_true = numpy.array([x[0] for x in y_true_and_pred])
            y_pred = numpy.array([x[1] for x in y_true_and_pred])

        elif ensemble_type == 'gradient-boosted-trees':
            # Loads the model
            model = GradientBoostedTreesModel.load(spark_context, model_filename)
            # Classifies the samples
            y_pred = model.predict(data.map(lambda x: x.features))
            # Merge ground-truth with predictions
            y_true_and_pred = data.map(lambda x: x.label).zip(y_pred).collect()
            y_true = numpy.array([x[0] for x in y_true_and_pred])
            y_pred = numpy.array([x[1] for x in y_true_and_pred])

        elif ensemble_type == 'extra-trees':
            # Loads the model
            with open(model_filename, 'rb') as f:
                model = pickle.load(f)
                f.close()
            # Retrieves all the dataset in the RDD to a local list
            local_data = data.collect()
            X = numpy.array([item.features   for item in local_data])
            y_true = numpy.array([item.label for item in local_data])
            del local_data
            # Classifies the samples
            y_pred = model.predict(X)
        else:
            raise Exception(f'{ensemble_type} is no a valid ensemble type')

        # Save results in text and graphically represented confusion matrices
        filename_prefix = f'{ensemble_type}-classification-results-{num_trees}-{impurity}-{max_depth}'
        save_results(results_dir, filename_prefix, y_true, y_pred)

    if do_prediction:
        if ensemble_type == 'random-forest':
            # Loads the model
            model = RandomForestModel.load(spark_context, model_filename)

        elif ensemble_type == 'gradient-boosted-trees':
            # Loads the model
            model = GradientBoostedTreesModel.load(spark_context, model_filename)

        elif ensemble_type == 'extra-trees':
            # Loads the model
            with open(model_filename, 'rb') as f:
                model = pickle.load(f)
                f.close()
        else:
            raise Exception(f'{ensemble_type} is no a valid ensemble type')

        #
        y_true_and_pred = list()
        #
        sliding_window_length = 5 * 60 # 1800 seconds -> 30 minutes
        sliding_window_step = 60 * 5 # 300 seconds -> 5 minute
        #
        target_class_accumulators = numpy.zeros(len(numpy.unique(label_mapping)))
        list_of_predictions = list()
        list_of_alarms = list()
        current_time = 0
        state = 'inter-ictal'
        
        old_patient = "non-existent-yet"
        #f = sys.stdin
        f = os.popen(f'hdfs dfs -cat {dataset_filename}')
        for line in f:
            parts = line.split(';')
            patient = parts[0]
            label =  int(parts[1])
            y_pred = model.predict(numpy.array([float(x) for x in parts[2:]]))
            y_pred = int(y_pred)
            #
            if patient != old_patient:
                # reset variables
                # 
                old_patient = patient
                target_class_accumulators = numpy.zeros(len(numpy.unique(label_mapping)))
                list_of_predictions = list()
                list_of_alarms = list()
                current_time = 0
                state = 'inter-ictal'
                for pred_label, pred_time in list_of_alarms:
                    y_true_and_pred.append((0, pred_label))
            #
            list_of_predictions.append(y_pred)
            target_class_accumulators[y_pred] += 1
            if len(list_of_predictions) > sliding_window_length:
                target_class_accumulators[list_of_predictions[0]] -= 1
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
            else:
                state = 'inter-ictal'
                #
                if current_time > 0 and current_time % sliding_window_step == 0:
                    if target_class_probs[1].sum() > 0.30:
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

        filename_prefix = f'{ensemble_type}-prediction-results-{num_trees}-{impurity}-{max_depth}'
        save_results(results_dir, filename_prefix, y_true, y_pred)

    spark_context.stop()
