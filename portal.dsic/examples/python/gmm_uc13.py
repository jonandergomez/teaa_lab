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
import sys
import time
import pickle
import math
import numpy

from pyspark import SparkContext
from pyspark.mllib.clustering import GaussianMixture, GaussianMixtureModel
from pyspark.mllib.stat.distribution import MultivariateGaussian

if __name__ == "__main__":
    sc = SparkContext(appName = "kmeans-uc13")  # SparkContext

    debug = 0
    filename = 'data/uc13.csv'
    list_of_n_components = list()
    #for n in range(  2,  10,   1): list_of_n_components.append(n)
    #for n in range( 10,  50,   5): list_of_n_components.append(n)
    #for n in range( 50, 100,  10): list_of_n_components.append(n)
    for n in range(100,  501, 100): list_of_n_components.append(n)
    num_partitions = 80

    for i in range(1, len(sys.argv)):
        if sys.argv[i] == '--filename':
            filename = sys.argv[i + 1]
        elif sys.argv[i] == '--n-partitions':
            num_partitions = int(sys.argv[i + 1])

    # Load and parse the data
    lines = sc.textFile(filename)
    data = lines.map(lambda line: numpy.array([float(x) for x in line.split(';')]))
    data = data.map(lambda x: x[1:]) # removes the first column corresponding to the label
    x = data.take(1)
    num_samples = data.count()
    dim = x[0].shape[0]

    print(f'loaded {num_samples} {dim}-dimensional samples into {data.getNumPartitions()} partitions')

    data.repartition(num_partitions)

    if debug > 1: print(type(x), x[0].shape)

    print(f'distributed {num_samples} {dim}-dimensional samples into {data.getNumPartitions()} partitions')

    ####################################################################
    # do standard scaling
    x_mean = data.reduce(lambda x1, x2: x1 + x2) / num_samples
    if debug > 0: print('mean:', x_mean)
    x_std = numpy.sqrt(data.map(lambda x: (x - x_mean) ** 2).reduce(lambda x1, x2: x1 + x2) / num_samples)
    if debug > 0: print('std:', x_std)
    data = data.map(lambda x: (x - x_mean) / x_std)
    ####################################################################

    data.persist()

    ####################################################################
    # Evaluate clustering by computing Within Set Sum of Squared Errors

    def compute_values_for_metrics(x):
        q = kmeans_model.predict(x)
        c_q = kmeans_model.centers[q]
        d = x - c_q
        sd = sum(d ** 2)
        return q, sd, math.sqrt(sd), numpy.outer(d, d)
    ####################################################################
    
    for n_components in list_of_n_components:

        # Build the model (cluster the data)
        starting_time = time.time()
        gmm_model = GaussianMixture.train(data, k = n_components,
                                            convergenceTol = 1.0e-3,
                                            maxIterations = 100,
                                            initialModel = None)
        with open('data/gmm_model-uc13-%03d.pkl' % n_components, 'wb') as f:
            gmm_params = list()
            gmm_params.append(gmm_model.k)
            gmm_params.append(gmm_model.weights)
            for g in gmm_model.gaussians:
                gmm_params.append(g.mu)
                gmm_params.append(g.sigma)
                print(type(g))
            pickle.dump(gmm_params, f)
            f.close()
        ending_time = time.time()
        print('processing time lapse for', n_components, 'clusters', ending_time - starting_time, 'seconds')

    ####################################################################
    data.unpersist()
    sc.stop()
