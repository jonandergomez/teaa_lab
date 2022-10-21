"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://personales.upv.es/jon)
    Version: 1.0
    Date: October 2022
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

    Using ensembles based on decision trees for classification or regression

    Ensemble types:
        Random Forest
        Gradient Boosted Trees
        Extremely Randomized Trees

    This code is only for Random Forest Classifiers, see the files
        ert_mnist.py for Extremely Randomdized Trees (or Extra Trees)
        gbt_mnist.py for Gradient Boosted Trees
"""

import sys
import os
import argparse
import numpy

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf

from sklearn.decomposition import PCA

from load_mnist import load_mnist
from utils_for_results import save_results


def main(args, sc):

    X, y = load_mnist()
    X /= 255.0
    print(X.shape, y.shape)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    #
    pca = PCA(n_components = args.pcaComponents if args.pcaComponents <= 1.0 else int(args.pcaComponents))
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    pcaComponents = X_train.shape[1]

    labels = numpy.unique(y_train)

    trainData = [[int(y_train[i]), Vectors.dense(X_train[i])] for i in range(len(X_train))]
    testData  = [[int(y_test[i]),  Vectors.dense(X_test[i])]  for i in range(len(X_test))]


    # Converting it to a DataFrame
    trainData = sc.createDataFrame(trainData, ['label', 'features'])
    testData  = sc.createDataFrame(testData,  ['label', 'features'])

    results_dir = f'{args.baseDir}/{args.resultsDir}'
    models_dir  = f'{args.baseDir}/{args.modelsDir}'
    log_dir     = f'{args.baseDir}/{args.logDir}'
    #os.makedirs(log_dir,     exist_ok = True)
    #os.makedirs(models_dir,  exist_ok = True)
    #os.makedirs(results_dir, exist_ok = True)

    for nt in args.numTrees.split(sep = ':'):
        if nt is None or len(nt) == 0: continue
        numTrees = int(nt)

        for md in args.maxDepth.split(sep = ':'):
            if md is None or len(md) == 0: continue
            maxDepth = int(md)

            # Creating the RandomForest model
            rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=numTrees, maxDepth=maxDepth)

            # Creating the pipeline, in this case with only one component, the model based on RandomForest 
            pipeline = Pipeline(stages=[rf])

            # Training the model
            model = pipeline.fit(trainData)

            if args.verbose > 0:
                # Printing some info about the model
                rfModel = model.stages[0] # [2]
                print(rfModel)  # summary only

            # TRAINING SUBSET
            # Make predictions
            predictions = model.transform(trainData)

            # Compute the test error
            evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
            accuracy = evaluator.evaluate(predictions)
            print("Train Error = %g" % (1.0 - accuracy))

            y_true_pred = predictions.select('label', 'prediction').collect()
            y_true = [y[0] for y in y_true_pred]
            y_pred = [y[1] for y in y_true_pred]

            filename_prefix = 'rf_%05d_pca_%04d_maxdepth_%03d' % (numTrees, pcaComponents, maxDepth)
            save_results(f'{results_dir}.train', filename_prefix = filename_prefix, y_true = y_true, y_pred = y_pred, elapsed_time = None, labels = labels)


            # TESTING SUBSET
            # Make predictions.
            predictions = model.transform(testData)

            # Select example rows to display
            if args.verbose > 0:
                predictions.select("label", "prediction", "features").show(5)

            # Compute the test error
            evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
            accuracy = evaluator.evaluate(predictions)
            print("Test Error = %g" % (1.0 - accuracy))

            y_true_pred = predictions.select('label', 'prediction').collect()
            y_true = [y[0] for y in y_true_pred]
            y_pred = [y[1] for y in y_true_pred]

            filename_prefix = 'rf_%05d_pca_%04d_maxdepth_%03d' % (numTrees, pcaComponents, maxDepth)
            save_results(f'{results_dir}.test', filename_prefix = filename_prefix, y_true = y_true, y_pred = y_pred, elapsed_time = None, labels = labels)
        # end for max depth
    # end for num trees

    sc.stop()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', default=0, type=int, help='Verbosity level')
    parser.add_argument('--numTrees', default="100:200", type=str, help='Colon separated list of number of trees in the Random Forest')
    parser.add_argument('--maxDepth', default="5:7", type=str, help='Colon separated list of the max depth of each tree in the Random Forest')
    parser.add_argument('--impurity', default="gini", type=str, help='Impurity type. Options are: gini or entropy')
    parser.add_argument('--pcaComponents', default=37, type=float, help='Number of components of PCA an integer > 1 or a float in the range [0,1[')
    parser.add_argument('--baseDir',    default='.',                type=str, help='Directory base from which create the directories for models, results and logs')
    parser.add_argument('--modelsDir',  default='models.l2.mnist',  type=str, help='Directory to save models --if it is the case')
    parser.add_argument('--resultsDir', default='results.l2.mnist', type=str, help='Directory where to store the results')
    parser.add_argument('--logDir',     default='log.l2.mnist',     type=str, help='Directory where to store the logs --if it is the case')

    sc = SparkSession.builder.appName(f"RandomForestClassifierForMNIST").getOrCreate()
    main(parser.parse_args(), sc)
    sc.stop()
