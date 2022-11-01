#
#
#
#
#
#

"""
Gradient Boosting Trees Regressor Example.
"""

import sys
import os
import argparse
import numpy

from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import RegressionEvaluator
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

    # Function to process each CSV line
    def process_csv_line(row):
        # returns a list with patient, time-to-seizure and features
        #return row[0], float(row[1]) / ratio, Vectors.dense([float(x) for x in row[2:]])
        return row[0], min(600.0, float(row[1])) / ratio, Vectors.dense([float(x) for x in row[2:]])

    # Processing each csv line in parallel and regenerate DataFrames with the appropriate column names
    trainData = trainData.rdd.map(process_csv_line).toDF(['patient', 'label', 'features'])
    testData  =  testData.rdd.map(process_csv_line).toDF(['patient', 'label', 'features'])

    #allData = trainData.unionAll(testData)

    if args.verbose > 0:
        trainData.show(5)
        testData.show(5)

        print(type(trainData), trainData.count())
        print(type(testData), testData.count())
        #print(type(allData), allData.count())

        if args.verbose > 1:
            trainData.describe().show()
            trainData.summary().show()
            testData.describe().show()
            testData.summary().show()
    

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
            gbt = GBTRegressor(labelCol="label", featuresCol="features", maxIter=numTrees, maxDepth=maxDepth, impurity=args.impurity)

            # Creating the pipeline, in this case with only one component, the model based on Gradient Boosted Trees
            pipeline = Pipeline(stages=[gbt])

            # Training the model
            model = pipeline.fit(trainData)

            # TRAINING SUBSET
            # Make predictions
            predictions = model.transform(trainData)

            # Select (prediction, true label) and compute test error
            evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
            rmse = evaluator.evaluate(predictions)
            print("Root Mean Squared Error (RMSE) on training data = %g" % rmse)

            def value_to_label(y):
                y = y * ratio
                if  -1.0 <= y <=   1.0: return 1
                elif 1.0 <= y <= 300.0: return 2
                elif        y <    0.0: return 3
                else:                   return 0

            y_true_pred = predictions.select('label', 'prediction').collect()
            y_true = [value_to_label(y[0]) for y in y_true_pred]
            y_pred = [value_to_label(y[1]) for y in y_true_pred]

            filename_prefix = 'gbt_%s_%05d_%03d_%s' % (args.patient, numTrees, maxDepth, 'pca' if args.usingPCA else 'no_pca')
            save_results(f'{results_dir}.train/gbt/{args.patient}', filename_prefix = filename_prefix, y_true = y_true, y_pred = y_pred, elapsed_time = None, labels = None)


            # TESTING SUBSET
            # Make predictions.
            predictions = model.transform(testData)

            # Select example rows to display
            if args.verbose > 0:
                predictions.select("label", "prediction", "features").show(5)

            # Select (prediction, true label) and compute test error
            evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
            rmse = evaluator.evaluate(predictions)
            print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

            y_true_pred = predictions.select('label', 'prediction').collect()
            y_true = [value_to_label(y[0]) for y in y_true_pred]
            y_pred = [value_to_label(y[1]) for y in y_true_pred]

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
    parser.add_argument('--modelsDir',  default='models.l2b.uc13',  type=str, help='Directory to save models --if it is the case')
    parser.add_argument('--resultsDir', default='results.l2b.uc13', type=str, help='Directory where to store the results')
    parser.add_argument('--logDir',     default='log.l2b.uc13',     type=str, help='Directory where to store the logs --if it is the case')

    sc = SparkSession.builder.appName("GradientBoostedTreesRegressorForUC13").getOrCreate()
    main(parser.parse_args(), sc)
    sc.stop()
