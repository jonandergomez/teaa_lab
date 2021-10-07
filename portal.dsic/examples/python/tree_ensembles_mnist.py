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

from sklearn.decomposition import PCA
from load_mnist import load_mnist
from utils_for_results import save_results

if __name__ == "__main__":
    """
    Usage: spark-submit --master local[4]  python/tree_ensembles_mnist.py --num-trees <nt> --max-depth <md> --impurity <imp> 
    """

    impurity = 'gini'
    max_depth = 10
    max_bins = 32
    num_trees = 100 # for Random Forest
    num_iterations = 100 # for Gradient Boosting
    
    ensemble_type = 'random-forest'

    verbose = 0

    spark_context = None
    num_partitions = 80
    models_dir = 'models.digits.2'
    log_dir = 'log.digits.2'
    results_dir = 'results.digits.2'
    do_training = False
    do_classification = False
    model_filename = None
    learning_rate = 0.1
    pca_components = 30
                                                   
    for i in range(len(sys.argv)):
        if   sys.argv[i] == "--verbosity"     :           verbose = int(sys.argv[i + 1])
        elif sys.argv[i] == "--num-partitions":    num_partitions = int(sys.argv[i + 1])
        elif sys.argv[i] == "--ensemble-type" :     ensemble_type = sys.argv[i + 1]
        elif sys.argv[i] == "--num-trees"     :         num_trees = int(sys.argv[i + 1])
        elif sys.argv[i] == "--num-iterations":    num_iterations = int(sys.argv[i + 1])
        elif sys.argv[i] == "--impurity"      :          impurity = sys.argv[i + 1]
        elif sys.argv[i] == "--max-depth"     :         max_depth = int(sys.argv[i + 1])
        elif sys.argv[i] == "--max-bins"      :          max_bins = int(sys.argv[i + 1])
        elif sys.argv[i] == "--learning-rate" :     learning_rate = float(sys.argv[i + 1])
        elif sys.argv[i] == "--model"         :    model_filename = sys.argv[i + 1]
        elif sys.argv[i] == "--train"         :       do_training = True
        elif sys.argv[i] == "--classify"      : do_classification = True
        elif sys.argv[i] == "--results-dir"   :       results_dir = sys.argv[i + 1]
        elif sys.argv[i] == "--models-dir"    :        models_dir = sys.argv[i + 1]
        elif sys.argv[i] == "--log-dir"       :           log_dir = sys.argv[i + 1]
        elif sys.argv[i] == "--pca"           :    pca_components = float(sys.argv[i + 1])


    if model_filename is None:
        model_filename = f'{models_dir}/{ensemble_type}-{num_trees}-{impurity}-{max_depth}'

    spark_context = SparkContext(appName = "TreeBasedEnsembles-dataset-MNIST")

    X, y = load_mnist()
    X /= 255.0
    print(X.shape, y.shape)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    #
    #pca = PCA(n_components = 0.95)
    if pca_components > 1: pca_components = int(pca_components)
    pca = PCA(n_components = pca_components)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    '''
        In order to use the Spark implementation of classifiers each data point must be
        an object of the class LabeledPoint.
        Basically, each object of this class has two attributes: label and features.
        For classification tasks, label should be an integer, for regression tasks label
        should be a floating-point number.
    '''
    label_mapping = [i for i in range(10)]
    label_unmapping = dict()
    if ensemble_type == 'gradient-boosted-trees':
        label_mapping[1] = 0
        label_mapping[7] = 100
        label_mapping[4] = 200
        label_mapping[5] = 300
        label_mapping[3] = 400
        label_mapping[2] = 500
        label_mapping[8] = 600
        label_mapping[6] = 700
        label_mapping[9] = 800
        label_mapping[0] = 900

    for i in range(len(label_mapping)):
        label_unmapping[label_mapping[i]] = i

    def remap_label(k):
        return label_mapping[k]

    def undo_remap_label(k):
        if label_mapping[0] == 0:
            return int(k)
        else:
            k = int(k / 100 + 0.5) * 100
            return label_unmapping[max(0, min(900, k))]
            
    rdd_train = spark_context.parallelize([LabeledPoint(remap_label(y), x.copy()) for x, y in zip(X_train, y_train)], numSlices = num_partitions)
    rdd_test  = spark_context.parallelize([LabeledPoint(remap_label(y), x.copy()) for x, y in zip(X_test, y_test)], numSlices = num_partitions)

    num_samples = rdd_train.count()

    print(f'train subset with {num_samples} distributed into {rdd_train.getNumPartitions()} partitions')
    print(f'test  subset with {rdd_test.count()} distributed into {rdd_test.getNumPartitions()} partitions')

    if do_training:
        if ensemble_type == 'random-forest':
            model = RandomForest.trainClassifier(rdd_train,
                                                numClasses = len(numpy.unique(y_train)),
                                                categoricalFeaturesInfo = {}, # nothing to use here
                                                numTrees = num_trees,
                                                featureSubsetStrategy = 'auto',
                                                impurity = impurity,
                                                maxDepth = max_depth,
                                                maxBins = max_bins)
            model.save(spark_context, model_filename)

        elif ensemble_type == 'gradient-boosted-trees':
            #model = GradientBoostedTrees.trainClassifier(rdd_train, loss = 'logLoss',
            model = GradientBoostedTrees.trainRegressor(rdd_train, loss = 'leastSquaresError',
                                                        categoricalFeaturesInfo = {}, # nothing to use here
                                                        learningRate = learning_rate,
                                                        numIterations = num_iterations,
                                                        maxDepth = max_depth,
                                                        maxBins = max_bins)
            model.save(spark_context, model_filename)

        elif ensemble_type == 'extra-trees':
            model = ExtraTreesClassifier(n_estimators = num_trees, criterion = impurity, max_depth = max_depth, n_jobs = -1, verbose = 1)
            model.fit(X_train, y_train)
            # Saves the model to local disk (in the master) using Pickle serialization for Python
            os.makedirs(models_dir, exist_ok = True)
            with open(model_filename, 'wb') as f:
                pickle.dump(model, f)
                f.close()
        else:
            raise Exception(f'{ensemble_type} is no a valid ensemble type')

            
    if do_classification:
        if ensemble_type in ['random-forest', 'gradient-boosted-trees']:
            if ensemble_type == 'random-forest':
                # Loads the model
                model = RandomForestModel.load(spark_context, model_filename)
            elif ensemble_type == 'gradient-boosted-trees':
                # Loads the model
                model = GradientBoostedTreesModel.load(spark_context, model_filename)
            #
            # Classifies the samples
            y_train_pred = model.predict(rdd_train.map(lambda x: x.features)).map(undo_remap_label)
            y_test_pred  = model.predict(rdd_test.map(lambda x: x.features)).map(undo_remap_label)
            # Merge ground-truth with predictions
            y_true_and_pred = rdd_train.map(lambda x: undo_remap_label(x.label)).zip(y_train_pred).collect()
            y_train_true = numpy.array([x[0] for x in y_true_and_pred])
            y_train_pred = numpy.array([x[1] for x in y_true_and_pred])
            #
            y_true_and_pred = rdd_test.map(lambda x: undo_remap_label(x.label)).zip(y_test_pred).collect()
            y_test_true = numpy.array([x[0] for x in y_true_and_pred])
            y_test_pred = numpy.array([x[1] for x in y_true_and_pred])

        elif ensemble_type == 'extra-trees':
            # Loads the model
            with open(model_filename, 'rb') as f:
                model = pickle.load(f)
                f.close()
            # Classifies the samples
            y_train_true = y_train
            y_train_pred = model.predict(X_train)
            y_test_true = y_test
            y_test_pred  = model.predict(X_test)
        else:
            raise Exception(f'{ensemble_type} is no a valid ensemble type')

        # Save results in text and graphically represented confusion matrices
        filename_prefix = f'{ensemble_type}-classification-results-{num_trees}-{impurity}-{max_depth}'
        save_results(f'{results_dir}.train', filename_prefix, y_train_true, y_train_pred)
        save_results(f'{results_dir}.test',  filename_prefix, y_test_true, y_test_pred)
    #
    spark_context.stop()
