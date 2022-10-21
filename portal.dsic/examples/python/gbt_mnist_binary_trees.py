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

    This code is only for Gradient Boosting Classifiers, see the files
        rf_mnist.py  for Ranfom Forest
        ert_mnist.py for Extremely Randomdized Trees (or Extra Trees)
"""

import sys
import os
import argparse
import numpy

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import GBTClassifier
#from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.feature import SQLTransformer
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

    def _l_vs_the_rest(label, l): return 0 if label == l else 1

    trainData = [[int(y_train[i]), Vectors.dense(X_train[i]),
                                _l_vs_the_rest(int(y_train[i]), 0),
                                _l_vs_the_rest(int(y_train[i]), 1),
                                _l_vs_the_rest(int(y_train[i]), 2),
                                _l_vs_the_rest(int(y_train[i]), 3),
                                _l_vs_the_rest(int(y_train[i]), 4),
                                _l_vs_the_rest(int(y_train[i]), 5),
                                _l_vs_the_rest(int(y_train[i]), 6),
                                _l_vs_the_rest(int(y_train[i]), 7),
                                _l_vs_the_rest(int(y_train[i]), 8),
                                _l_vs_the_rest(int(y_train[i]), 9) ] for i in range(len(X_train))]
    testData  = [[int(y_test[i]),  Vectors.dense(X_test[i]),
                                _l_vs_the_rest(int(y_test[i]), 0),
                                _l_vs_the_rest(int(y_test[i]), 1),
                                _l_vs_the_rest(int(y_test[i]), 2),
                                _l_vs_the_rest(int(y_test[i]), 3),
                                _l_vs_the_rest(int(y_test[i]), 4),
                                _l_vs_the_rest(int(y_test[i]), 5),
                                _l_vs_the_rest(int(y_test[i]), 6),
                                _l_vs_the_rest(int(y_test[i]), 7),
                                _l_vs_the_rest(int(y_test[i]), 8),
                                _l_vs_the_rest(int(y_test[i]), 9) ] for i in range(len(X_test))]

    # Converting it to a DataFrame
    trainData = sc.createDataFrame(trainData, ['label', 'features', 'l0', 'l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9'])
    testData  = sc.createDataFrame(testData,  ['label', 'features', 'l0', 'l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9'])

    _0_vs_the_rest = SQLTransformer(statement="SELECT * FROM __THIS__")
    _1_vs_the_rest = SQLTransformer(statement="SELECT * FROM __THIS__ WHERE label > 0")
    _2_vs_the_rest = SQLTransformer(statement="SELECT * FROM __THIS__ WHERE label > 1")
    _3_vs_the_rest = SQLTransformer(statement="SELECT * FROM __THIS__ WHERE label > 2")
    _4_vs_the_rest = SQLTransformer(statement="SELECT * FROM __THIS__ WHERE label > 3")
    _5_vs_the_rest = SQLTransformer(statement="SELECT * FROM __THIS__ WHERE label > 4")
    _6_vs_the_rest = SQLTransformer(statement="SELECT * FROM __THIS__ WHERE label > 5")
    _7_vs_the_rest = SQLTransformer(statement="SELECT * FROM __THIS__ WHERE label > 6")
    _8_vs_the_rest = SQLTransformer(statement="SELECT * FROM __THIS__ WHERE label > 7")
    #_9_vs_the_rest = SQLTransformer(statement="SELECT features, l9 FROM __THIS__ WHERE label > 8")

    postprocessLabelsAndPredictions_0 = SQLTransformer(statement="SELECT features, label, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, (prediction + 0) as prediction FROM __THIS__")
    postprocessLabelsAndPredictions_1 = SQLTransformer(statement="SELECT features, label, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, (prediction + 1) as prediction FROM __THIS__")
    postprocessLabelsAndPredictions_2 = SQLTransformer(statement="SELECT features, label, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, (prediction + 2) as prediction FROM __THIS__")
    postprocessLabelsAndPredictions_3 = SQLTransformer(statement="SELECT features, label, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, (prediction + 3) as prediction FROM __THIS__")
    postprocessLabelsAndPredictions_4 = SQLTransformer(statement="SELECT features, label, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, (prediction + 4) as prediction FROM __THIS__")
    postprocessLabelsAndPredictions_5 = SQLTransformer(statement="SELECT features, label, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, (prediction + 5) as prediction FROM __THIS__")
    postprocessLabelsAndPredictions_6 = SQLTransformer(statement="SELECT features, label, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, (prediction + 6) as prediction FROM __THIS__")
    postprocessLabelsAndPredictions_7 = SQLTransformer(statement="SELECT features, label, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, (prediction + 7) as prediction FROM __THIS__")
    postprocessLabelsAndPredictions_8 = SQLTransformer(statement="SELECT features, label, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, (prediction + 8) as prediction FROM __THIS__")

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

            print('numTrees', numTrees, 'maxDepth', maxDepth, 'impurity', args.impurity)

            # Creating the RandomForest model
            gbt_0 = GBTClassifier(labelCol="l0", featuresCol="features", maxIter=numTrees, maxDepth=maxDepth, impurity=args.impurity)
            gbt_1 = GBTClassifier(labelCol="l1", featuresCol="features", maxIter=numTrees, maxDepth=maxDepth, impurity=args.impurity)
            gbt_2 = GBTClassifier(labelCol="l2", featuresCol="features", maxIter=numTrees, maxDepth=maxDepth, impurity=args.impurity)
            gbt_3 = GBTClassifier(labelCol="l3", featuresCol="features", maxIter=numTrees, maxDepth=maxDepth, impurity=args.impurity)
            gbt_4 = GBTClassifier(labelCol="l4", featuresCol="features", maxIter=numTrees, maxDepth=maxDepth, impurity=args.impurity)
            gbt_5 = GBTClassifier(labelCol="l5", featuresCol="features", maxIter=numTrees, maxDepth=maxDepth, impurity=args.impurity)
            gbt_6 = GBTClassifier(labelCol="l6", featuresCol="features", maxIter=numTrees, maxDepth=maxDepth, impurity=args.impurity)
            gbt_7 = GBTClassifier(labelCol="l7", featuresCol="features", maxIter=numTrees, maxDepth=maxDepth, impurity=args.impurity)
            gbt_8 = GBTClassifier(labelCol="l8", featuresCol="features", maxIter=numTrees, maxDepth=maxDepth, impurity=args.impurity)
            #gbt_9 = GBTClassifier(labelCol="l9", featuresCol="features", maxIter=numTrees, maxDepth=maxDepth, impurity=args.impurity)

            # Creating the pipelines for training
            pipeline_0 = Pipeline(stages=[_0_vs_the_rest, gbt_0, postprocessLabelsAndPredictions_0])
            pipeline_1 = Pipeline(stages=[_1_vs_the_rest, gbt_1, postprocessLabelsAndPredictions_1])
            pipeline_2 = Pipeline(stages=[_2_vs_the_rest, gbt_2, postprocessLabelsAndPredictions_2])
            pipeline_3 = Pipeline(stages=[_3_vs_the_rest, gbt_3, postprocessLabelsAndPredictions_3])
            pipeline_4 = Pipeline(stages=[_4_vs_the_rest, gbt_4, postprocessLabelsAndPredictions_4])
            pipeline_5 = Pipeline(stages=[_5_vs_the_rest, gbt_5, postprocessLabelsAndPredictions_5])
            pipeline_6 = Pipeline(stages=[_6_vs_the_rest, gbt_6, postprocessLabelsAndPredictions_6])
            pipeline_7 = Pipeline(stages=[_7_vs_the_rest, gbt_7, postprocessLabelsAndPredictions_7])
            pipeline_8 = Pipeline(stages=[_8_vs_the_rest, gbt_8, postprocessLabelsAndPredictions_8])

            # Training the model
            model_0 = pipeline_0.fit(trainData)
            model_1 = pipeline_1.fit(trainData)
            model_2 = pipeline_2.fit(trainData)
            model_3 = pipeline_3.fit(trainData)
            model_4 = pipeline_4.fit(trainData)
            model_5 = pipeline_5.fit(trainData)
            model_6 = pipeline_6.fit(trainData)
            model_7 = pipeline_7.fit(trainData)
            model_8 = pipeline_8.fit(trainData)

            model_for_inference_0 = PipelineModel(stages=model_0.stages[1:])
            model_for_inference_1 = PipelineModel(stages=model_1.stages[1:])
            model_for_inference_2 = PipelineModel(stages=model_2.stages[1:])
            model_for_inference_3 = PipelineModel(stages=model_3.stages[1:])
            model_for_inference_4 = PipelineModel(stages=model_4.stages[1:])
            model_for_inference_5 = PipelineModel(stages=model_5.stages[1:])
            model_for_inference_6 = PipelineModel(stages=model_6.stages[1:])
            model_for_inference_7 = PipelineModel(stages=model_7.stages[1:])
            model_for_inference_8 = PipelineModel(stages=model_8.stages[1:])


            # TRAINING SUBSET
            # Make predictions
            predictions_from_0 = model_for_inference_0.transform(trainData)
            predictions_from_1 = model_for_inference_1.transform(predictions_from_0.filter('prediction > 0').drop('prediction'))
            predictions_from_2 = model_for_inference_2.transform(predictions_from_1.filter('prediction > 1').drop('prediction'))
            predictions_from_3 = model_for_inference_3.transform(predictions_from_2.filter('prediction > 2').drop('prediction'))
            predictions_from_4 = model_for_inference_4.transform(predictions_from_3.filter('prediction > 3').drop('prediction'))
            predictions_from_5 = model_for_inference_5.transform(predictions_from_4.filter('prediction > 4').drop('prediction'))
            predictions_from_6 = model_for_inference_6.transform(predictions_from_5.filter('prediction > 5').drop('prediction'))
            predictions_from_7 = model_for_inference_7.transform(predictions_from_6.filter('prediction > 6').drop('prediction'))
            predictions_from_8 = model_for_inference_8.transform(predictions_from_7.filter('prediction > 7').drop('prediction'))

            predictions = predictions_from_0.filter('prediction = 0')
            predictions = predictions.union(predictions_from_1.filter('prediction = 1'))
            predictions = predictions.union(predictions_from_2.filter('prediction = 2'))
            predictions = predictions.union(predictions_from_3.filter('prediction = 3'))
            predictions = predictions.union(predictions_from_4.filter('prediction = 4'))
            predictions = predictions.union(predictions_from_5.filter('prediction = 5'))
            predictions = predictions.union(predictions_from_6.filter('prediction = 6'))
            predictions = predictions.union(predictions_from_7.filter('prediction = 7'))
            predictions = predictions.union(predictions_from_8)

            # Compute the test error
            evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
            accuracy = evaluator.evaluate(predictions)
            print("Train Error = %g" % (1.0 - accuracy))

            y_true_pred = predictions.select('label', 'prediction').collect()
            y_true = [y[0] for y in y_true_pred]
            y_pred = [min(9, max(0, int(y[1]))) for y in y_true_pred]

            filename_prefix = 'gbt_%05d_pca_%04d_maxdepth_%03d' % (numTrees, pcaComponents, maxDepth)
            save_results(f'{results_dir}.train', filename_prefix = filename_prefix, y_true = y_true, y_pred = y_pred, elapsed_time = None, labels = labels)


            # TESTING SUBSET
            # Make predictions.
            predictions_from_0 = model_for_inference_0.transform(testData)
            predictions_from_1 = model_for_inference_1.transform(predictions_from_0.filter('prediction > 0').drop('prediction'))
            predictions_from_2 = model_for_inference_2.transform(predictions_from_1.filter('prediction > 1').drop('prediction'))
            predictions_from_3 = model_for_inference_3.transform(predictions_from_2.filter('prediction > 2').drop('prediction'))
            predictions_from_4 = model_for_inference_4.transform(predictions_from_3.filter('prediction > 3').drop('prediction'))
            predictions_from_5 = model_for_inference_5.transform(predictions_from_4.filter('prediction > 4').drop('prediction'))
            predictions_from_6 = model_for_inference_6.transform(predictions_from_5.filter('prediction > 5').drop('prediction'))
            predictions_from_7 = model_for_inference_7.transform(predictions_from_6.filter('prediction > 6').drop('prediction'))
            predictions_from_8 = model_for_inference_8.transform(predictions_from_7.filter('prediction > 7').drop('prediction'))

            predictions = predictions_from_0.filter('prediction == 0')
            predictions = predictions.union(predictions_from_1.filter('prediction = 1'))
            predictions = predictions.union(predictions_from_2.filter('prediction = 2'))
            predictions = predictions.union(predictions_from_3.filter('prediction = 3'))
            predictions = predictions.union(predictions_from_4.filter('prediction = 4'))
            predictions = predictions.union(predictions_from_5.filter('prediction = 5'))
            predictions = predictions.union(predictions_from_6.filter('prediction = 6'))
            predictions = predictions.union(predictions_from_7.filter('prediction = 7'))
            predictions = predictions.union(predictions_from_8)

            # Select example rows to display
            if args.verbose > 0:
                predictions.select("label", "prediction", "features").show(5)

            # Compute the test error
            evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
            accuracy = evaluator.evaluate(predictions)
            print("Test Error = %g" % (1.0 - accuracy))

            y_true_pred = predictions.select('label', 'prediction').collect()
            y_true = [y[0] for y in y_true_pred]
            y_pred = [min(9, max(0, int(y[1]))) for y in y_true_pred]

            filename_prefix = 'gbt_%05d_pca_%04d_maxdepth_%03d' % (numTrees, pcaComponents, maxDepth)
            save_results(f'{results_dir}.test', filename_prefix = filename_prefix, y_true = y_true, y_pred = y_pred, elapsed_time = None, labels = labels)
        # end for max depth
    # end for num trees

    sc.stop()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', default=0, type=int, help='Verbosity level')
    parser.add_argument('--numTrees', default="100:200", type=str, help='Colon separated list of number of trees in the Random Forest')
    parser.add_argument('--maxDepth', default="5:7", type=str, help='Colon separated list of the max depth of each tree in the Random Forest')
    parser.add_argument('--impurity', default="variance", type=str, help='Impurity type. Valid options is variance, because we have to use a regressor')
    parser.add_argument('--pcaComponents', default=37, type=float, help='Number of components of PCA an integer > 1 or a float in the range [0,1[')
    parser.add_argument('--baseDir',    default='.',                type=str, help='Directory base from which create the directories for models, results and logs')
    parser.add_argument('--modelsDir',  default='models.l2b.mnist',  type=str, help='Directory to save models --if it is the case')
    parser.add_argument('--resultsDir', default='results.l2b.mnist', type=str, help='Directory where to store the results')
    parser.add_argument('--logDir',     default='log.l2b.mnist',     type=str, help='Directory where to store the logs --if it is the case')

    sc = SparkSession.builder.appName(f"GradientBoostedTreeForMNIST").getOrCreate()
    main(parser.parse_args(), sc)
    sc.stop()
