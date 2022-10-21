#
#
#
#
#
#

"""
Random Forest Classifier Example.
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
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from utils_for_results import save_results


def main(args, sc):

    # Loading and parsing the data file, converting it to a DataFrame.
    if args.usingPCA:
        trainData = sc.read.option('delimiter', ';').csv(f"data/uc13/uc13-{args.patient}-21x20-train-pca.csv")
        testData  = sc.read.option('delimiter', ';').csv(f"data/uc13/uc13-{args.patient}-21x20-test-pca.csv")
    else:
        trainData = sc.read.option('delimiter', ';').csv(f"data/uc13/uc13-{args.patient}-21x20-train.csv")
        testData  = sc.read.option('delimiter', ';').csv(f"data/uc13/uc13-{args.patient}-21x20-test.csv")

    # Preparing the labels according to the number of target classes
    #   2 for binary classification
    #  10 for multi-class classification
    label_mapping = [i for i in range(10)]
    if args.doBinaryClassification:
        label_mapping = [0 for i in range(10)]
        label_mapping[1] = 1
    labels = numpy.unique(label_mapping)
        
    # Function to process each CSV line
    def process_csv_line(row):
        # returns a list with patient, label and features
        return row[0], int(label_mapping[int(row[1])]), Vectors.dense([float(x) for x in row[2:]])

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

    # Creating the RandomForest model
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=args.numTrees)

    # Creating the pipeline, in this case with only one component, the model based on RandomForest 
    pipeline = Pipeline(stages=[rf])

    # Training the model
    model = pipeline.fit(trainData)

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

    filename_prefix = 'rf_%s_%05d_%s' % (args.patient, args.numTrees, 'pca' if args.usingPCA else 'no_pca')
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

    filename_prefix = 'rf_%s_%05d_%s' % (args.patient, args.numTrees, 'pca' if args.usingPCA else 'no_pca')
    save_results(f'{results_dir}.test', filename_prefix = filename_prefix, y_true = y_true, y_pred = y_pred, elapsed_time = None, labels = labels)

    sc.stop()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('patient', type=str, help='Patient identifier')
    ''' For future versions
    parser.add_argument('--doBinaryClassification', default=False, type=bool,
                            action=argparse.BooleanOptionalAction,
                            help='Flag to indicate whether do the binary classification, default is false')
    '''
    parser.add_argument('--doBinaryClassification', action='store_true')
    parser.add_argument('--no-doBinaryClassification', action='store_false')
    parser.set_defaults(doBinaryClassification = False)
    parser.add_argument('--usingPCA', action='store_true')
    parser.add_argument('--no-usingPCA', action='store_false')
    parser.set_defaults(usingPCA = False)
    parser.add_argument('--verbose', default=0, type=int, help='Verbosity level')
    parser.add_argument('--numTrees', default=100, type=int, help='Number of trees in the Random Forest')

    parser.add_argument('--baseDir',    default='.',               type=str, help='Directory base from which create the directories for models, results and logs')
    parser.add_argument('--modelsDir',  default='models.l2.uc13',  type=str, help='Directory to save models --if it is the case')
    parser.add_argument('--resultsDir', default='results.l2.uc13', type=str, help='Directory where to store the results')
    parser.add_argument('--logDir',     default='log.l2.uc13',     type=str, help='Directory where to store the logs --if it is the case')

    sc = SparkSession.builder.appName("RandomForestClassifierForUC13").getOrCreate()
    main(parser.parse_args(), sc)
    sc.stop()
