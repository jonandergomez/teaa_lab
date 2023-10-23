#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Code modified from the Spark sample code to adapt it for subject
#
#       14009 "Scalable Machine Learning Techniques"
#
#   Bachelor's degree in Data Science
#   School of Informatics  (http://www.etsinf.upv.es)
#   Technical University of Valencia (http://www.upv.es)
#
#
import os
import sys
import time
import pickle
import math
import numpy
import tempfile

import machine_learning
from utils_for_results import save_results
from eeg_load_data import load_csv_from_uc13

use_spark = True
try:
    from pyspark import SparkContext
    from pyspark.mllib.clustering import KMeans, KMeansModel
except:
    use_spark = False


# --------------------------------------------------------------------------------
def estimate_gmm(data, spark_context, max_components, models_dir = None, log_dir = None, batch_size = 500):
    num_partitions = data.getNumPartitions()
    K = (data.count() + batch_size - 1) / batch_size 
    K = ((K // num_partitions) + 1) * num_partitions
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

    temp_dirname = tempfile.mkdtemp(prefix = 'mle', suffix = '.gmm')

    mle = machine_learning.MLE( covar_type = covar_type,
                                dim = dim_x,
                                log_dir = log_dir if log_dir is not None else temp_dirname,
                                models_dir = models_dir if models_dir is not None else temp_dirname)
    mle.fit_with_spark( spark_context = spark_context,
                        samples = samples,
                        max_components = max_components,
                        epsilon = 1.0e-4)
    samples.unpersist()
    #
    return mle.gmm
# ------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    spark = SparkContext(appName = "gmm-uc13-21x14")  # SparkContext

    hdfs_url = 'hdfs://teaa-master-ubuntu22:8020'

    verbose = 0
    num_channels = 21
    base_dir = '.'
    patient = 'chb02'
    data_format = 'pca136'
    list_of_gmm_components = [10] # [10, 12, 15, 20, 25, 30]
    covar_type = 'diagonal'
    num_partitions = 60
    do_train = False
    do_classify = False
    do_binary_classification = False

    for i in range(1, len(sys.argv)):
        if   sys.argv[i] == '--patient'                 : patient = sys.argv[i + 1]
        elif sys.argv[i] == '--data-format'             : data_format = sys.argv[i + 1]
        elif sys.argv[i] == '--models-dir'              : models_dir = sys.argv[i + 1]
        elif sys.argv[i] == '--n-partitions'            : num_partitions = int(sys.argv[i + 1])
        elif sys.argv[i] == '--gmm-components'          : list_of_gmm_components = [int(s) for s in sys.argv[i + 1].split(sep = ',')]
        elif sys.argv[i] == '--covar-type'              : covar_type = sys.argv[i + 1]
        elif sys.argv[i] == "--covar"                   : covar_type = sys.argv[i + 1]
        elif sys.argv[i] == '--do-binary-classification': do_binary_classification = True
        elif sys.argv[i] == '-v'                        : verbose += 1
        #elif sys.argv[i] == '--train'                   : do_train = True
        #elif sys.argv[i] == '--classify'                : do_classify = True

    do_z_transform = (data_format == '21x14') # It should be False if PCA was applied

    models_dir  = f'{base_dir}/models/uc13/gmm'
    log_dir     = f'{base_dir}/logs/uc13/gmm'
    results_dir = f'{base_dir}/results/uc13/gmm/{patient}'

    task = 'binary-classification' if do_binary_classification else 'multi-class-classification'

    os.makedirs(log_dir,     exist_ok = True)
    os.makedirs(models_dir,  exist_ok = True)
    os.makedirs(results_dir, exist_ok = True)

    print('sizes of GMM', list_of_gmm_components)

    if patient == 'ALL':
        train_filenames = [f'{hdfs_url}/data/uc13/21x14/uc13-chb{i:02d}-{data_format}-time-to-seizure.csv' for i in range(1,17)]
        test_filenames  = [f'{hdfs_url}/data/uc13/21x14/uc13-chb{i:02d}-{data_format}-time-to-seizure.csv' for i in range(17,25)]
    else:
        train_filenames = [f'{hdfs_url}/data/uc13/21x14/uc13-{patient}-{data_format}-time-to-seizure-train.csv']
        test_filenames  = [f'{hdfs_url}/data/uc13/21x14/uc13-{patient}-{data_format}-time-to-seizure-test.csv']

    # Loads and repartitions the data
    rdd_train = load_csv_from_uc13(spark, train_filenames, num_partitions, do_binary_classification = do_binary_classification)
    rdd_test  = load_csv_from_uc13(spark,  test_filenames, num_partitions, do_binary_classification = do_binary_classification)

    # BEGIN: Perform the standard scalation
    if do_z_transform:
        mean = rdd_train.map(lambda sample: sample[3]).reduce(lambda x, y: x + y) / rdd_train.count()
        variance = rdd_train.map(lambda sample: (sample[3] - mean)**2).reduce(lambda x, y: x + y) / rdd_train.count()
        sigma = numpy.sqrt(variance)
        #
        rdd_train = rdd_train.map(lambda sample: (sample[0], sample[1], sample[2], (sample[3] - mean) / sigma))
        rdd_test  =  rdd_test.map(lambda sample: (sample[0], sample[1], sample[2], (sample[3] - mean) / sigma))
    # END: Perform the standard scalation

    #print(rdd_train.first())
    #print(rdd_test.first())
    print(rdd_train.count(), rdd_train.getNumPartitions(), rdd_test.count(), rdd_test.getNumPartitions())


    if do_binary_classification:
        labels = [0, 1]
    else:
        l1 = rdd_train.map(lambda sample: (sample[2], 1)).reduceByKey(lambda x, y: x + y).collect()
        l2 = rdd_test.map(lambda sample: (sample[2], 1)).reduceByKey(lambda x, y: x + y).collect()
        labels = [x[0] for x in (l1 + l2)]
        labels = list(numpy.unique(labels))

    print(labels)


    for gmm_components in list_of_gmm_components:
        gmm_filename = f'{models_dir}/gmm-{patient}-{data_format}-{task}-{gmm_components:04d}.pkl'

        # do GMM MLE for each target class
        if os.path.exists(gmm_filename):
            with open(gmm_filename, 'rb') as f:
                model = pickle.load(f)
                f.close()
        else:
            model = dict()
            for label in labels:
                data = rdd_train.filter(lambda x: x[2] == label).map(lambda x: x[3])
                print(f'working with {data.count()} samples from target class {label} distributed in {data.getNumPartitions()} partitions')
                gmm = None
                if data.count() > 1000:
                    gmm = estimate_gmm(data = data, spark_context = spark, max_components = gmm_components)
                elif data.count() > 100:
                    gmm = estimate_gmm(data = data, spark_context = spark, max_components = 2)
                elif data.count() > 0:
                    gmm = estimate_gmm(data = data, spark_context = spark, max_components = 1)
                #
                model[label] = gmm
            #
            with open(gmm_filename, 'wb') as f:
                pickle.dump(model, f)
                f.close()
        
        #target_classes_a_priori_probabilities = accumulators.sum(axis = 1) / accumulators.sum()

        def classify_sample(t):
            patient, tts, label, x = t
            _probs = list()
            for i in labels:
                gmm = model[i]
                if gmm is not None:
                    _log_densities = gmm.log_densities(x, with_a_priori_probs = True)
                    _densities = numpy.exp(_log_densities - _log_densities.max()) * numpy.exp(_log_densities.max())
                    _probs.append(_densities.sum())
                else:
                    _probs.append(0.0)
            # for
            _probs = numpy.array(_probs)
            return (patient, label, _probs.argmax())

        data_rdds = {'train' : rdd_train, 'test' : rdd_test}

        for subset in ['train', 'test']:
            y_true_and_pred = data_rdds[subset].map(classify_sample).collect()
            y_true = numpy.array([x[1] for x in y_true_and_pred])
            y_pred = numpy.array([x[2] for x in y_true_and_pred])

            filename_prefix = f'gmm-{patient}-{data_format}-{task}-{gmm_components:04d}'
            save_results(f'{results_dir}/{subset}', filename_prefix, y_true, y_pred, labels = labels)

    spark.stop()
