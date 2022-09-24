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
#
# Code modified from the Spark sample code to adapt it for subject
#
#       14009 "Scalable Machine Learning Techniques"
#
#   Bachelor's degree in Data Science
#   School of Informatics  (http://www.etsinf.upv.es)
#   Technical University of Valencia (http://www.upv.es)
#


import sys
import time
from operator import add

from pyspark.sql import SparkSession
from pyspark.sql import DataFrameWriter

starting_time = time.time()

def measure_time(msg = None):
    global starting_time
    t = time.time()
    if msg is None:
        print(f'time lapse since last time measurement {t - starting_time} seconds')
    else:
        print(f'{msg}: {t - starting_time} seconds')
    starting_time = t


do_the_collect_into_the_driver = False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: word_count_2 <file>", file=sys.stderr)
        sys.exit(-1)

    measure_time('excution starts')

    spark = SparkSession\
        .builder\
        .appName("PythonWordCount")\
        .getOrCreate()

    measure_time('creating the Spark environment')

    reviews_df = spark.read.json(sys.argv[1])

    measure_time('loading data')

    reviews_df.printSchema()

    measure_time('printing the schema')

    reviews_df.createOrReplaceTempView("reviews")

    measure_time('creating the temporal view')

    review_lines = spark.sql("SELECT review_text from reviews")

    measure_time('performing the SQL-like select')

    print('num samples:', review_lines.count(), 'num partitions', review_lines.rdd.getNumPartitions())
    measure_time('counting samples/lines')

    # to see what an object in the data frame is, a Row
    print(review_lines.first())


    def clean_strings(s):
        if s is None: return ''

        o = ''
        for c in s.lower():
            if ord('A') <= ord(c) <= ord('z'):
                o += c
            elif ord(' ') <= ord(c) <= 127:
                o += ' '
        return o

    # transformation, returns an RDD of the after using only the first colum of each row
    review_lines = review_lines.rdd.flatMap(lambda row: row[0].split(' ')) # use row[0] to access the text

    # transformation, returns an RDD of the same data type by removing None objects and empty strings
    review_lines = review_lines.filter(lambda x: x is not None and len(x.strip()) > 0)

    # transformation, returns an RDD of the same data type by splitting with the ' as delimiter
    review_lines = review_lines.flatMap(lambda x: x.split("'"))

    # data cleasing
    review_lines = review_lines.map(clean_strings)

    # transformation, returns an RDD of the same data type by splitting with the white space as delimiter
    review_lines = review_lines.flatMap(lambda x: x.split(' '))

    # transformation, returns an RDD of the same data type by removing None objects and empty strings
    review_lines = review_lines.filter(lambda x: x is not None and len(x.strip()) > 0)

    # transformation, returns an RDD of where each word is paired with an integer equal to 1
    review_lines = review_lines.map(lambda x: (x, 1))

    # transformation, accumulate by key and returns a new RDD with tuples (key, counter)
    counts = review_lines.reduceByKey(lambda x, y: x + y) #(add)

    # transformation, returns an RDD of the same data type by removing less frequent words
    counts = counts.filter(lambda x: x[1] >= 10) # comment this line to get all the words

    measure_time('couting word occurences')

    # the lazy execution of all, because reduceByKey() is a transformation
    # and not an action, makes that the time after the collect() includes
    # all the previous transformations, so the time measurement when working
    # with Spark RDD must be considered appropriately

    if do_the_collect_into_the_driver:
        output = counts.collect()
        measure_time('retrieving counts from cluster to the driver program')
        output.sort(key = lambda x: x[0], reverse = False) # sort alphabetically
        measure_time('sorting 1')
        output.sort(key = lambda x: x[1], reverse = True) #  sort by counter in descending order
        measure_time('sorting 2')

        with open('word_count_2.out', 'wt') as f:
            for (word, count) in output:
                print("%s: %i" % (word, count), file = f)
            f.close()
    else:
        df = counts.toDF()
        measure_time('converting to DataFrame')
        df.write.csv(path = 'word_count_2.csv', mode = 'overwrite', sep = ';')

    measure_time('saving')

    spark.stop()
