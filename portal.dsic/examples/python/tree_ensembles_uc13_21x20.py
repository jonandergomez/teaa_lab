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

from utils_for_results import save_results

from pyspark import SparkContext


if __name__ == "__main__":
    """
    Usage: spark-submit --master local[4]  python/tree_ensembles_uc13.py --dataset data/uc13-train.csv --num-trees <nt> --max-depth <md> --impurity <imp> 
    """

    impurity = 'gini'
    max_depth = 10
    max_bins = 32
    num_trees = 100 # for Random Forest
    num_iterations = 100 # for Gradient Boosting

    home_dir = os.getenv('HOME')
    if home_dir is None:
        raise Exception("Impossible to continue without the user's home directory")

    f = os.popen('hostname')
    hostname = f.readline().strip()
    f.close()
    if hostname is not None and hostname.find('eibds01') >= 0:
        hdfs_home_dir = '/user/cluster'
    else:
        hdfs_home_dir = '/user/ubuntu'
    
    ensemble_type = 'random-forest'

    verbose = 0
    model_filename = None

    spark_context = None
    num_partitions = 80
    num_channels = 21
    global_patient = 'chb01'
    do_training = False
    do_classification = False
    do_prediction = False
    learning_rate = 0.1
    label_mapping = [i for i in range(10)]
    subset = 'train'
                                                   
    for i in range(len(sys.argv)):
        #if   sys.argv[i] == "--dataset"       :  dataset_filename = sys.argv[i + 1]
        #elif sys.argv[i] == "--results-dir"   :       results_dir = sys.argv[i + 1]
        #elif sys.argv[i] == "--models-dir"    :        models_dir = sys.argv[i + 1]
        #elif sys.argv[i] == "--log-dir"       :           log_dir = sys.argv[i + 1]
        if sys.argv[i] in ['--dataset', '--results-dir', '--models-dir', '--log-dir']:
            print()
            print('Option no longer accepted, for this code use --patient --subset')
            print()
            sys.exit(0)

        elif sys.argv[i] == "--subset"        :            subset = sys.argv[i + 1]
        elif sys.argv[i] == "--patient"       :    global_patient = sys.argv[i + 1]
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

    results_dir = f'{home_dir}/uc13-21x20/{global_patient}/results.2'
    models_dir = f'uc13-21x20/{global_patient}/models.2'
    log_dir = f'{home_dir}/uc13-21x20/{global_patient}/log.2'
    dataset_filename = f'{hdfs_home_dir}/data/uc13/uc13-{global_patient}-21x20-{subset}.csv'

    if model_filename is None:
        model_filename = f'{models_dir}/{ensemble_type}-{num_trees}-{impurity}-{max_depth}'

    spark_context = SparkContext(appName = "Ensemble-of-Trees-dataset-UC13")


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

    def csv_line_to_labeled_points(line):
        parts = line.split(';')
        label = label_mapping[int(parts[1])]
        sample = numpy.array([float(x) for x in parts[2:]]).reshape(21, -1)
        return [LabeledPoint(label, x) for x in sample]

    os.makedirs(log_dir,    exist_ok = True)
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
    samples = csv_lines.flatMap(csv_line_to_labeled_points)
    # Retrives basic info from the RDD object
    num_samples = samples.count()
    x = samples.take(1)
    dim = x[0].features.shape[0]

    print(f'loaded {num_samples} {dim}-dimensional samples into {samples.getNumPartitions()} partitions')


    if do_training:
        if ensemble_type == 'random-forest':
            model = RandomForest.trainClassifier(samples,
                                                numClasses = len(numpy.unique(label_mapping)),
                                                categoricalFeaturesInfo = {}, # nothing to use here
                                                numTrees = num_trees,
                                                featureSubsetStrategy = 'auto',
                                                impurity = impurity,
                                                maxDepth = max_depth,
                                                maxBins = max_bins)
            model.save(spark_context, model_filename)

        elif ensemble_type == 'gradient-boosted-trees':
            model = GradientBoostedTrees.trainClassifier(samples,
                                                        categoricalFeaturesInfo = {}, # nothing to use here
                                                        loss = 'logLoss',
                                                        learningRate = learning_rate,
                                                        numIterations = num_iterations,
                                                        maxDepth = max_depth,
                                                        maxBins = max_bins)
            model.save(spark_context, model_filename)

        elif ensemble_type == 'extra-trees':
            # Retrieves all the dataset in the RDD to a local list
            local_data = samples.collect()
            X = numpy.array([item.features for item in local_data]) # get features
            y = numpy.array([item.label    for item in local_data]) # get labels
            del local_data # free memory, just in case
            #
            model = ExtraTreesClassifier(n_estimators = num_trees, criterion = impurity, max_depth = max_depth, n_jobs = -1, verbose = 1)
            model.fit(X, y)
            # Saves the model to local disk (in the master) using Pickle serialization for Python
            os.makedirs(f'{home_dir}/{models_dir}', exist_ok = True)
            with open(f'{home_dir}/{model_filename}', 'wb') as f:
                pickle.dump(model, f)
                f.close()
        else:
            raise Exception(f'{ensemble_type} is no a valid ensemble type')

            
    if do_classification:
        if ensemble_type in ['random-forest', 'gradient-boosted-trees']:
            # Loads the model
            if ensemble_type == 'random-forest':
                model = RandomForestModel.load(spark_context, model_filename)
            else:
                model = GradientBoostedTreesModel.load(spark_context, model_filename)
            #
            # Classifies the samples
            y_pred = model.predict(data.map(lambda x: x.features.reshape(num_channels, -1)[0])).map(lambda p: numpy.eye(10)[int(p)])
            for ch in range(1, num_channels):
                pred = model.predict(data.map(lambda x: x.features.reshape(num_channels, -1)[ch])).map(lambda p: numpy.eye(10)[int(p)])
                y_pred = y_pred.zip(pred).map(lambda x: x[0] + x[1])
            #
            y_pred = y_pred.map(lambda x: x.argmax())
            # 
            # Merge ground-truth with predictions
            y_true_and_pred = data.map(lambda x: x.label).zip(y_pred).collect()
            y_true = numpy.array([x[0] for x in y_true_and_pred])
            y_pred = numpy.array([x[1] for x in y_true_and_pred])

        elif ensemble_type == 'extra-trees':
            # Loads the model
            with open(f'{home_dir}/{model_filename}', 'rb') as f:
                model = pickle.load(f)
                f.close()
            # Retrieves all the dataset in the RDD to a local list
            local_data = data.collect()
            X = numpy.array([item.features   for item in local_data])
            y_true = numpy.array([item.label for item in local_data])
            del local_data
            # Classifies the samples
            y_pred = model.predict(X.reshape(-1, 20)).astype(int)
            print(sum(y_pred == 0), sum(y_pred == 1))
            y_pred = numpy.eye(10)[y_pred]
            y_pred = y_pred.reshape(-1, num_channels, 10).sum(axis = 1).argmax(axis = 1)
        else:
            raise Exception(f'{ensemble_type} is no a valid ensemble type')

        # Save results in text and graphically represented confusion matrices
        filename_prefix = f'{ensemble_type}-classification-results-{num_trees}-{impurity}-{max_depth}'
        save_results(f'{results_dir}-{subset}', filename_prefix, y_true, y_pred)

    if do_prediction:
        if ensemble_type == 'random-forest':
            # Loads the model
            model = RandomForestModel.load(spark_context, model_filename)

        elif ensemble_type == 'gradient-boosted-trees':
            # Loads the model
            model = GradientBoostedTreesModel.load(spark_context, model_filename)

        elif ensemble_type == 'extra-trees':
            # Loads the model
            with open(f'{home_dir}/{model_filename}', 'rb') as f:
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
        save_results(f'{results_dir}-{subset}', filename_prefix, y_true, y_pred)

    spark_context.stop()
