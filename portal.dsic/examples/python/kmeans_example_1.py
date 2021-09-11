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
import sys
import time
import pickle
# $example on$
from numpy import array
from math import sqrt
# $example off$

from pyspark import SparkContext
# $example on$
from pyspark.mllib.clustering import KMeans, KMeansModel
# $example off$

if __name__ == "__main__":
    sc = SparkContext(appName = "kmeans-example")  # SparkContext

    filename = 'data/vectors_train_1000000_30_10.csv'
    k = 10

    for i in range(1, len(sys.argv)):
        if sys.argv[i] == '--filename':
            filename = sys.argv[i + 1]
        elif sys.argv[i] == '--n-clusters':
            k = int(sys.argv[i + 1])

    starting_time = time.time()
    # $example on$
    # Load and parse the data
    lines = sc.textFile(filename)
    data = lines.map(lambda line: array([float(x) for x in line.split(';')]))

    data.repartition(80)

    x = data.take(1)
    #print(type(x), x[0].shape)
    n = data.count()
    d = x[0].shape[0]

    print(f'loaded {n} {d}-dimensional samples')

    # Build the model (cluster the data)
    kmeans_model = KMeans.train(data, k, maxIterations = 20) #, initializationMode = "random")

    # Evaluate clustering by computing Within Set Sum of Squared Errors
    def error(point):
        center = kmeans_model.centers[kmeans_model.predict(point)]
        return sqrt(sum([x**2 for x in (point - center)]))

    WSSSE = data.map(lambda point: error(point)).reduce(lambda x, y: x + y)
    print("Within Set Sum of Squared Error = " + str(WSSSE) + ' normalized ' + str(WSSSE / n))

    # Save and load model
    #kmeans_model.save(sc, "data/kmeans_model")
    #sameModel = KMeansModel.load(sc, "data/kmeans_model")

    print(len(kmeans_model.centers), kmeans_model.centers[0].shape)

    with open('data/kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans_model.centers, f)
        f.close()

    with open('data/kmeans_model.pkl', 'rb') as f:
        centers = pickle.load(f)
        f.close()
    print(len(centers), centers[0].shape)
    # $example off$

    ending_time = time.time()

    sc.stop()

    print('processing time lapse', ending_time - starting_time, 'seconds')
