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
    X = X / 255.0
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

    def _l_vs_the_rest_(label, l): return 0 if label == l else 1
    def _s1_vs_s2_(label, s1): return 0 if label in s1 else 1

    trainData = [[int(y_train[i]), Vectors.dense(X_train[i]),
                                _l_vs_the_rest_(int(y_train[i]), 0),
                                _l_vs_the_rest_(int(y_train[i]), 1),
                                _l_vs_the_rest_(int(y_train[i]), 2),
                                _l_vs_the_rest_(int(y_train[i]), 3),
                                _l_vs_the_rest_(int(y_train[i]), 4),
                                _l_vs_the_rest_(int(y_train[i]), 5),
                                _l_vs_the_rest_(int(y_train[i]), 6),
                                _l_vs_the_rest_(int(y_train[i]), 7),
                                _l_vs_the_rest_(int(y_train[i]), 8),
                                _l_vs_the_rest_(int(y_train[i]), 9),
                                _s1_vs_s2_(int(y_train[i]), (3, 5, 8),
                                _s1_vs_s2_(int(y_train[i]), (3, 8),
                                _s1_vs_s2_(int(y_train[i]), (4, 9)] for i in range(len(X_train))]
    testData  = [[int(y_test[i]),  Vectors.dense(X_test[i]),
                                _l_vs_the_rest_(int(y_test[i]), 0),
                                _l_vs_the_rest_(int(y_test[i]), 1),
                                _l_vs_the_rest_(int(y_test[i]), 2),
                                _l_vs_the_rest_(int(y_test[i]), 3),
                                _l_vs_the_rest_(int(y_test[i]), 4),
                                _l_vs_the_rest_(int(y_test[i]), 5),
                                _l_vs_the_rest_(int(y_test[i]), 6),
                                _l_vs_the_rest_(int(y_test[i]), 7),
                                _l_vs_the_rest_(int(y_test[i]), 8),
                                _l_vs_the_rest_(int(y_test[i]), 9),
                                _s1_vs_s2_(int(y_test[i], (3, 5, 8),
                                _s1_vs_s2_(int(y_test[i]), (3, 8),
                                _s1_vs_s2_(int(y_test[i]), (4, 9)] for i in range(len(X_test))]

    # Converting it to a DataFrame
    trainData = sc.createDataFrame(trainData, ['label', 'features', 'l0', 'l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9', 'l358', 'l38', 'l49']).repartition(600)
    testData  = sc.createDataFrame(testData,  ['label', 'features', 'l0', 'l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9', 'l358', 'l38', 'l49']).repartition(600)

    _0_vs_the_rest = SQLTransformer(statement="SELECT * FROM __THIS__")
    _1_vs_the_rest = SQLTransformer(statement="SELECT * FROM __THIS__ WHERE label > 0")
    _2_vs_the_rest = SQLTransformer(statement="SELECT * FROM __THIS__ WHERE label > 1")
    _6_vs_the_rest = SQLTransformer(statement="SELECT * FROM __THIS__ WHERE label > 2")
    _358_vs_479_   = SQLTransformer(statement="SELECT * FROM __THIS__ WHERE label in [3, 5, 8, 4, 7, 9]")
    _38_vs_5_      = SQLTransformer(statement="SELECT * FROM __THIS__ WHERE label in [3, 5, 8]")
    _3_vs_8_       = SQLTransformer(statement="SELECT * FROM __THIS__ WHERE label in [3, 8]")
    _49_vs_7_      = SQLTransformer(statement="SELECT * FROM __THIS__ WHERE label in [4, 7, 9]")
    _4_vs_9_       = SQLTransformer(statement="SELECT * FROM __THIS__ WHERE label in [4, 9]")

    postprocessLabelsAndPredictions = SQLTransformer(statement="SELECT features, label, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l358, l38, l49, prediction FROM __THIS__")
    #postprocessLabelsAndPredictions_0   = SQLTransformer(statement="SELECT features, label, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l358, l38, l49, prediction FROM __THIS__")
    #postprocessLabelsAndPredictions_1   = SQLTransformer(statement="SELECT features, label, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l358, l38, l49, prediction FROM __THIS__")
    #postprocessLabelsAndPredictions_2   = SQLTransformer(statement="SELECT features, label, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l358, l38, l49, prediction FROM __THIS__")
    #postprocessLabelsAndPredictions_6   = SQLTransformer(statement="SELECT features, label, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l358, l38, l49, prediction FROM __THIS__")
    #postprocessLabelsAndPredictions_358 = SQLTransformer(statement="SELECT features, label, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l358, l38, l49, prediction FROM __THIS__")
    #postprocessLabelsAndPredictions_38  = SQLTransformer(statement="SELECT features, label, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l358, l38, l49, prediction FROM __THIS__")
    #postprocessLabelsAndPredictions_3   = SQLTransformer(statement="SELECT features, label, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l358, l38, l49, prediction FROM __THIS__")
    #postprocessLabelsAndPredictions_49  = SQLTransformer(statement="SELECT features, label, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l358, l38, l49, prediction FROM __THIS__")
    #postprocessLabelsAndPredictions_4   = SQLTransformer(statement="SELECT features, label, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l358, l38, l49, prediction FROM __THIS__")

    results_dir = f'{args.baseDir}/{args.resultsDir}'
    models_dir  = f'{args.baseDir}/{args.modelsDir}'
    log_dir     = f'{args.baseDir}/{args.logDir}'
    #os.makedirs(log_dir,     exist_ok = True)
    #os.makedirs(models_dir,  exist_ok = True)
    os.makedirs(results_dir, exist_ok = True)

    for nt in args.numTrees.split(sep = ':'):
        if nt is None or len(nt) == 0: continue
        numTrees = int(nt)

        for md in args.maxDepth.split(sep = ':'):
            if md is None or len(md) == 0: continue
            maxDepth = int(md)

            print('numTrees', numTrees, 'maxDepth', maxDepth, 'impurity', args.impurity)

            # Creating the RandomForest model
            gbt_0          = GBTClassifier(labelCol="l0",   featuresCol="features", maxIter=numTrees, maxDepth=maxDepth, impurity=args.impurity)
            gbt_1          = GBTClassifier(labelCol="l1",   featuresCol="features", maxIter=numTrees, maxDepth=maxDepth, impurity=args.impurity)
            gbt_2          = GBTClassifier(labelCol="l2",   featuresCol="features", maxIter=numTrees, maxDepth=maxDepth, impurity=args.impurity)
            gbt_6          = GBTClassifier(labelCol="l6",   featuresCol="features", maxIter=numTrees, maxDepth=maxDepth, impurity=args.impurity)
            gbt_358_vs_479 = GBTClassifier(labelCol="l358", featuresCol="features", maxIter=numTrees, maxDepth=maxDepth, impurity=args.impurity)
            gbt_38_vs_5    = GBTClassifier(labelCol="l38",  featuresCol="features", maxIter=numTrees, maxDepth=maxDepth, impurity=args.impurity)
            gbt_3_vs_8     = GBTClassifier(labelCol="l3",   featuresCol="features", maxIter=numTrees, maxDepth=maxDepth, impurity=args.impurity)
            gbt_49_vs_7    = GBTClassifier(labelCol="l49",  featuresCol="features", maxIter=numTrees, maxDepth=maxDepth, impurity=args.impurity)
            gbt_4_vs_9     = GBTClassifier(labelCol="l4",   featuresCol="features", maxIter=numTrees, maxDepth=maxDepth, impurity=args.impurity)

            # Creating the pipelines for training
            pipeline_1 = Pipeline(stages=[_0_vs_the_rest, gbt_0,          postprocessLabelsAndPredictions])
            pipeline_2 = Pipeline(stages=[_1_vs_the_rest, gbt_1,          postprocessLabelsAndPredictions])
            pipeline_3 = Pipeline(stages=[_2_vs_the_rest, gbt_2,          postprocessLabelsAndPredictions])
            pipeline_4 = Pipeline(stages=[_6_vs_the_rest, gbt_6,          postprocessLabelsAndPredictions])
            pipeline_5 = Pipeline(stages=[_358_vs_479_,   gbt_358_vs_479, postprocessLabelsAndPredictions])
            pipeline_6 = Pipeline(stages=[_38_vs_5_,      gbt_38_vs_5,    postprocessLabelsAndPredictions])
            pipeline_7 = Pipeline(stages=[_3_vs_8_,       gbt_3_vs_8,     postprocessLabelsAndPredictions])
            pipeline_8 = Pipeline(stages=[_49_vs_7_,      gbt_49_vs_7,    postprocessLabelsAndPredictions])
            pipeline_9 = Pipeline(stages=[_4_vs_9_,       gbt_4_vs_9,     postprocessLabelsAndPredictions])

            # Training the model
            model_1 = pipeline_1.fit(trainData)
            model_2 = pipeline_2.fit(trainData)
            model_3 = pipeline_3.fit(trainData)
            model_4 = pipeline_4.fit(trainData)
            model_5 = pipeline_5.fit(trainData)
            model_6 = pipeline_6.fit(trainData)
            model_7 = pipeline_7.fit(trainData)
            model_8 = pipeline_8.fit(trainData)
            model_9 = pipeline_9.fit(trainData)

            model_for_inference_1 = PipelineModel(stages=model_1.stages[1:])
            model_for_inference_2 = PipelineModel(stages=model_2.stages[1:])
            model_for_inference_3 = PipelineModel(stages=model_3.stages[1:])
            model_for_inference_4 = PipelineModel(stages=model_4.stages[1:])
            model_for_inference_5 = PipelineModel(stages=model_5.stages[1:])
            model_for_inference_6 = PipelineModel(stages=model_6.stages[1:])
            model_for_inference_7 = PipelineModel(stages=model_7.stages[1:])
            model_for_inference_8 = PipelineModel(stages=model_8.stages[1:])
            model_for_inference_9 = PipelineModel(stages=model_9.stages[1:])


            # TRAINING SUBSET
            # Make predictions
            predictions_from_1 = model_for_inference_1.transform(trainData)                                                      #=> {0}     labelled as 0, {1,2,3,4,5,6,7,8,9} as 1
            predictions_from_2 = model_for_inference_2.transform(predictions_from_1.filter('prediction = 1').drop('prediction')) #=> {1}     labelled as 0, {2,3,4,5,6,7,8,9} as 1
            predictions_from_3 = model_for_inference_3.transform(predictions_from_2.filter('prediction = 1').drop('prediction')) #=> {2}     labelled as 0, {3,4,5,6,7,8,9} as 1
            predictions_from_4 = model_for_inference_4.transform(predictions_from_3.filter('prediction = 1').drop('prediction')) #=> {6}     labelled as 0, {3,4,5,7,8,9} as 1
            predictions_from_5 = model_for_inference_5.transform(predictions_from_4.filter('prediction = 1').drop('prediction')) #=> {3,5,8} labelled as 0, {4,7,9} as 1

            predictions_from_6 = model_for_inference_6.transform(predictions_from_5.filter('prediction = 0').drop('prediction')) #=> {3,8}   labelled as 0, {5} as 1
            predictions_from_7 = model_for_inference_7.transform(predictions_from_6.filter('prediction = 0').drop('prediction')) #=> {3}     labelled as 0, {8} as 1
            predictions_from_8 = model_for_inference_8.transform(predictions_from_4.filter('prediction = 1').drop('prediction')) #=> {4,9}   labelled as 0, {7} as 1
            predictions_from_9 = model_for_inference_9.transform(predictions_from_8.filter('prediction = 0').drop('prediction')) #=> {4}     labelled as 0, {9} as 1

            predictions = predictions_from_1.filter('prediction = 0')                       # adds predictions for label 0
            df = predictions_from_2.filter('prediction = 0')
            predictions = predictions.union(df.withColumn('prediction', df.prediction + 1)) # adds predictions for label 1
            df = predictions_from_3.filter('prediction = 0')
            predictions = predictions.union(df.withColumn('prediction', df.prediction + 2)) # adds predictions for label 2
            df = predictions_from_4.filter('prediction = 0')
            predictions = predictions.union(df.withColumn('prediction', df.prediction + 6)) # adds predictions for label 6
            df = predictions_from_6.filter('prediction = 1')
            predictions = predictions.union(df.withColumn('prediction', df.prediction + 4)) # adds predictions for label 5
            df = predictions_from_7.filter('prediction = 0')
            predictions = predictions.union(df.withColumn('prediction', df.prediction + 3)) # adds predictions for label 3
            df = predictions_from_7.filter('prediction = 1')
            predictions = predictions.union(df.withColumn('prediction', df.prediction + 7)) # adds predictions for label 8
            df = predictions_from_8.filter('prediction = 1')
            predictions = predictions.union(df.withColumn('prediction', df.prediction + 6)) # adds predictions for label 7
            df = predictions_from_9.filter('prediction = 0')
            predictions = predictions.union(df.withColumn('prediction', df.prediction + 4)) # adds predictions for label 4
            df = predictions_from_9.filter('prediction = 1')
            predictions = predictions.union(df.withColumn('prediction', df.prediction + 8)) # adds predictions for label 9


            # Compute the error in training
            evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
            accuracy = evaluator.evaluate(predictions)
            print("Train Error = %g" % (1.0 - accuracy))

            y_true_pred = predictions.select('label', 'prediction').collect()
            y_true = [y[0] for y in y_true_pred]
            y_pred = [min(9, max(0, int(y[1]))) for y in y_true_pred]

            filename_prefix = 'gbt_%05d_pca_%04d_maxdepth_%03d' % (numTrees, pcaComponents, maxDepth)
            save_results(f'{results_dir}/gbt.b/train', filename_prefix = filename_prefix, y_true = y_true, y_pred = y_pred, elapsed_time = None, labels = labels)


            # TESTING SUBSET
            # Make predictions.
            predictions_from_1 = model_for_inference_1.transform(testData)                                                       #=> {0}     labelled as 0, {1,2,3,4,5,6,7,8,9} as 1
            predictions_from_2 = model_for_inference_2.transform(predictions_from_1.filter('prediction = 1').drop('prediction')) #=> {1}     labelled as 0, {2,3,4,5,6,7,8,9} as 1
            predictions_from_3 = model_for_inference_3.transform(predictions_from_2.filter('prediction = 1').drop('prediction')) #=> {2}     labelled as 0, {3,4,5,6,7,8,9} as 1
            predictions_from_4 = model_for_inference_4.transform(predictions_from_3.filter('prediction = 1').drop('prediction')) #=> {6}     labelled as 0, {3,4,5,7,8,9} as 1
            predictions_from_5 = model_for_inference_5.transform(predictions_from_4.filter('prediction = 1').drop('prediction')) #=> {3,5,8} labelled as 0, {4,7,9} as 1

            predictions_from_6 = model_for_inference_6.transform(predictions_from_5.filter('prediction = 0').drop('prediction')) #=> {3,8}   labelled as 0, {5} as 1
            predictions_from_7 = model_for_inference_7.transform(predictions_from_6.filter('prediction = 0').drop('prediction')) #=> {3}     labelled as 0, {8} as 1
            predictions_from_8 = model_for_inference_8.transform(predictions_from_4.filter('prediction = 1').drop('prediction')) #=> {4,9}   labelled as 0, {7} as 1
            predictions_from_9 = model_for_inference_9.transform(predictions_from_8.filter('prediction = 0').drop('prediction')) #=> {4}     labelled as 0, {9} as 1

            predictions = predictions_from_1.filter('prediction = 0')                       # adds predictions for label 0
            df = predictions_from_2.filter('prediction = 0')
            predictions = predictions.union(df.withColumn('prediction', df.prediction + 1)) # adds predictions for label 1
            df = predictions_from_3.filter('prediction = 0')
            predictions = predictions.union(df.withColumn('prediction', df.prediction + 2)) # adds predictions for label 2
            df = predictions_from_4.filter('prediction = 0')
            predictions = predictions.union(df.withColumn('prediction', df.prediction + 6)) # adds predictions for label 6
            df = predictions_from_6.filter('prediction = 1')
            predictions = predictions.union(df.withColumn('prediction', df.prediction + 4)) # adds predictions for label 5
            df = predictions_from_7.filter('prediction = 0')
            predictions = predictions.union(df.withColumn('prediction', df.prediction + 3)) # adds predictions for label 3
            df = predictions_from_7.filter('prediction = 1')
            predictions = predictions.union(df.withColumn('prediction', df.prediction + 7)) # adds predictions for label 8
            df = predictions_from_8.filter('prediction = 1')
            predictions = predictions.union(df.withColumn('prediction', df.prediction + 6)) # adds predictions for label 7
            df = predictions_from_9.filter('prediction = 0')
            predictions = predictions.union(df.withColumn('prediction', df.prediction + 4)) # adds predictions for label 4
            df = predictions_from_9.filter('prediction = 1')
            predictions = predictions.union(df.withColumn('prediction', df.prediction + 8)) # adds predictions for label 9

            # Select example rows to display
            if args.verbose > 0:
                predictions.select("label", "prediction", "features").show(5)

            # Compute the error in test
            evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
            accuracy = evaluator.evaluate(predictions)
            print("Test Error = %g" % (1.0 - accuracy))

            y_true_pred = predictions.select('label', 'prediction').collect()
            y_true = [y[0] for y in y_true_pred]
            y_pred = [min(9, max(0, int(y[1]))) for y in y_true_pred]

            filename_prefix = 'gbt_%05d_pca_%04d_maxdepth_%03d' % (numTrees, pcaComponents, maxDepth)
            save_results(f'{results_dir}/gbt.b/test', filename_prefix = filename_prefix, y_true = y_true, y_pred = y_pred, elapsed_time = None, labels = labels)
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
    parser.add_argument('--modelsDir',  default='models/digits/ensembles',  type=str, help='Directory to save models --if it is the case')
    parser.add_argument('--resultsDir', default='results/digits/ensembles', type=str, help='Directory where to store the results')
    parser.add_argument('--logDir',     default='logs/digits/ensembles',     type=str, help='Directory where to store the logs --if it is the case')

    sc = SparkSession.builder.appName(f"GradientBoostedTreeForMNIST").getOrCreate()
    main(parser.parse_args(), sc)
    sc.stop()
