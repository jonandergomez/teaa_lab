#
#
#
#
#
#

"""
Gradient Boosting Trees Cascade of Binary Classifiers Example.
"""

import sys
import os
import argparse
import numpy

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.feature import SQLTransformer
from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from utils_for_results import save_results

#ratio = 3600.0
ratio = 1.0

def main(args, sc):

    # Loading and parsing the data file, converting it to a DataFrame.
    if args.usingPCA:
        trainData = sc.read.option('delimiter', ';').csv(f"data/uc13/uc13-{args.patient}-21x20-time-to-seizure-train-pca.csv")
        testData  = sc.read.option('delimiter', ';').csv(f"data/uc13/uc13-{args.patient}-21x20-time-to-seizure-test-pca.csv")
    else:
        trainData = sc.read.option('delimiter', ';').csv(f"data/uc13/uc13-{args.patient}-21x20-time-to-seizure-train.csv")
        testData  = sc.read.option('delimiter', ';').csv(f"data/uc13/uc13-{args.patient}-21x20-time-to-seizure-test.csv")

    # class 0 -> all the samples far from 10 minutes (600 secons) from the beginning of a seizure
    # class 1 -> all the samples in the period from 0 seconds to 10 minutes (600 secons) after the end of a seizure
    # class 2 -> all the samples previous to a seizure, in this case less than 10 minutes 
    # class 3 -> all the samples within seizures
    def _partition_0(tts): return 0 if tts >= 300 else 1 # assings 0 to samples of target class 0 and 1 to the rest
    def _partition_1(tts): return 0 if tts <    0 else 1 # assings 0 to samples of target class 1 and 1 to samples of target classes 2 and 3
    def _partition_2(tts): return 0 if tts >    0 else 1 # assings 0 to samples of target class 2 and 1 to samples of target class 3

    # Function to process each CSV line
    def process_csv_line(row):
        # returns a list with patient, time-to-seizure and features
        #return row[0], float(row[1]) / ratio, Vectors.dense([float(x) for x in row[2:]])
        tts = float(row[1])
        tts = min(300, max(-300, tts))
        l0 = _partition_0(tts)
        l1 = _partition_1(tts)
        l2 = _partition_2(tts)
        label = l0 + l0 * l1 + l0 * l1 * l2
        return row[0], tts, label, l0, l1, l2, Vectors.dense([float(x) for x in row[2:]])

    # Processing each csv line in parallel and regenerate DataFrames with the appropriate column names
    trainData = trainData.rdd.map(process_csv_line).toDF(['patient', 'time-to-seizure', 'label', 'l0', 'l1', 'l2', 'features'])
    testData  =  testData.rdd.map(process_csv_line).toDF(['patient', 'time-to-seizure', 'label', 'l0', 'l1', 'l2', 'features'])

    print(numpy.unique(trainData.select('label').collect()))
    print(numpy.unique(testData.select('label').collect()))


    _0_vs_the_rest = SQLTransformer(statement="SELECT * FROM __THIS__")              # does not filter, all training samples will be used
    _1_vs_the_rest = SQLTransformer(statement="SELECT * FROM __THIS__ WHERE l0 > 0") # removes samples that belong to target class 0
    _2_vs_the_rest = SQLTransformer(statement="SELECT * FROM __THIS__ WHERE l1 > 0") # removes samples that belong to target classes 0 and 1

    postprocessLabelsAndPredictions_0 = SQLTransformer(statement="SELECT features, label, l0, l1, l2, (prediction + 0) as prediction FROM __THIS__")
    postprocessLabelsAndPredictions_1 = SQLTransformer(statement="SELECT features, label, l0, l1, l2, (prediction + 1) as prediction FROM __THIS__")
    postprocessLabelsAndPredictions_2 = SQLTransformer(statement="SELECT features, label, l0, l1, l2, (prediction + 2) as prediction FROM __THIS__")

    results_dir = f'{args.baseDir}/{args.resultsDir}'
    models_dir  = f'{args.baseDir}/{args.modelsDir}'
    log_dir     = f'{args.baseDir}/{args.logDir}'
    #os.makedirs(log_dir,     exist_ok = True)
    #os.makedirs(models_dir,  exist_ok = True)
    #os.makedirs(results_dir, exist_ok = True)

    trainData.persist()
    testData.persist()

    for nt in args.numTrees.split(sep = ':'):
        if nt is None or len(nt) == 0: continue
        numTrees = int(nt)

        for md in args.maxDepth.split(sep = ':'):
            if md is None or len(md) == 0: continue
            maxDepth = int(md)

            print('patient', args.patient, 'numTrees', numTrees, 'maxDepth', maxDepth, 'impurity', args.impurity)

            # Creating the Gradient Boosted Trees model
            gbt_0 = GBTClassifier(labelCol="l0", featuresCol="features", maxIter=numTrees, maxDepth=maxDepth, impurity=args.impurity)
            gbt_1 = GBTClassifier(labelCol="l1", featuresCol="features", maxIter=numTrees, maxDepth=maxDepth, impurity=args.impurity)
            gbt_2 = GBTClassifier(labelCol="l2", featuresCol="features", maxIter=numTrees, maxDepth=maxDepth, impurity=args.impurity)

            # Creating the pipeline, in this case with only one component, the model based on Gradient Boosted Trees
            pipeline_0 = Pipeline(stages=[_0_vs_the_rest, gbt_0, postprocessLabelsAndPredictions_0])
            pipeline_1 = Pipeline(stages=[_1_vs_the_rest, gbt_1, postprocessLabelsAndPredictions_1])
            pipeline_2 = Pipeline(stages=[_2_vs_the_rest, gbt_2, postprocessLabelsAndPredictions_2])

            # Training the model
            model_0 = pipeline_0.fit(trainData)
            model_1 = pipeline_1.fit(trainData)
            model_2 = pipeline_2.fit(trainData)

            model_for_inference_0 = PipelineModel(stages=model_0.stages[1:])
            model_for_inference_1 = PipelineModel(stages=model_1.stages[1:])
            model_for_inference_2 = PipelineModel(stages=model_2.stages[1:])


            # TRAINING SUBSET
            # Make predictions
            predictions_from_0 = model_for_inference_0.transform(trainData)
            predictions_from_1 = model_for_inference_1.transform(predictions_from_0.filter('prediction > 0').drop('prediction'))
            predictions_from_2 = model_for_inference_2.transform(predictions_from_1.filter('prediction > 1').drop('prediction'))

            predictions = predictions_from_0.filter('prediction = 0')
            predictions = predictions.union(predictions_from_1.filter('prediction = 1'))
            predictions = predictions.union(predictions_from_2)

            # Select (prediction, true label) and compute error on training set
            evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
            accuracy = evaluator.evaluate(predictions)
            print("Train Error = %g" % (1.0 - accuracy))

            y_true_pred = predictions.select('label', 'prediction').collect()
            y_true = [y[0] for y in y_true_pred]
            y_pred = [int(y[1]) for y in y_true_pred]

            filename_prefix = 'gbt_%s_%05d_%03d_%s' % (args.patient, numTrees, maxDepth, 'pca' if args.usingPCA else 'no_pca')
            save_results(f'{results_dir}.train/gbt/{args.patient}', filename_prefix = filename_prefix, y_true = y_true, y_pred = y_pred, elapsed_time = None, labels = None)


            # TESTING SUBSET
            # Make predictions
            predictions_from_0 = model_for_inference_0.transform(testData)
            predictions_from_1 = model_for_inference_1.transform(predictions_from_0.filter('prediction > 0').drop('prediction'))
            predictions_from_2 = model_for_inference_2.transform(predictions_from_1.filter('prediction > 1').drop('prediction'))

            predictions = predictions_from_0.filter('prediction = 0')
            predictions = predictions.union(predictions_from_1.filter('prediction = 1'))
            predictions = predictions.union(predictions_from_2)

            # Select (prediction, true label) and compute error on testing set
            evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
            accuracy = evaluator.evaluate(predictions)
            print("Test Error = %g" % (1.0 - accuracy))

            y_true_pred = predictions.select('label', 'prediction').collect()
            y_true = [y[0] for y in y_true_pred]
            y_pred = [int(y[1]) for y in y_true_pred]

            # Select example rows to display
            if args.verbose > 0:
                predictions.select("label", "prediction", "features").show(5)

            save_results(f'{results_dir}.test/gbt/{args.patient}', filename_prefix = filename_prefix, y_true = y_true, y_pred = y_pred, elapsed_time = None, labels = None)
        # end for max depth
    # end for num trees

    trainData.unpersist()
    testData.unpersist()

    sc.stop()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('patient', type=str, help='Patient identifier')
    parser.add_argument('--usingPCA', action='store_true')
    parser.add_argument('--no-usingPCA', action='store_false')
    parser.set_defaults(usingPCA = False)
    parser.add_argument('--verbose', default=0, type=int, help='Verbosity level')
    parser.add_argument('--numTrees', default="10:20:30:50:70:100:200", type=str, help='Colon separated list of number of trees in the Gradient Boosted Trees')
    parser.add_argument('--maxDepth', default="3:5:7:9:11:13", type=str, help='Colon separated list of the max depth of each tree in the Gradient Boosted Trees')
    parser.add_argument('--impurity', default="variance", type=str, help='Impurity type. Valid options is variance, because we have to use a regressor')

    parser.add_argument('--baseDir',    default='.',               type=str, help='Directory base from which create the directories for models, results and logs')
    parser.add_argument('--modelsDir',  default='models.l2c.uc13',  type=str, help='Directory to save models --if it is the case')
    parser.add_argument('--resultsDir', default='results.l2c.uc13', type=str, help='Directory where to store the results')
    parser.add_argument('--logDir',     default='log.l2c.uc13',     type=str, help='Directory where to store the logs --if it is the case')

    sc = SparkSession.builder.appName("GradientBoostedTreesBinaryTreesForUC13").getOrCreate()
    main(parser.parse_args(), sc)
    sc.stop()
