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


import os
import sys
from operator import add

from pyspark.sql import SparkSession


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: word_count_1 <file>", file=sys.stderr)
        sys.exit(-1)

    spark = SparkSession\
        .builder\
        .appName("PythonWordCount")\
        .getOrCreate()

    lines = spark.read.text(sys.argv[1])
    lines = lines.rdd.map(lambda r: r[0])
    #print(lines.take(10))

    def clean_strings(s):
        if s is None: return ''

        o = ''
        for c in s.lower():
            if c in ' abcdefghijklmnopqrstuvwxyz':
                o += c
        return o

    lines = lines.map(clean_strings)

    counts = lines.flatMap(lambda x: x.split(' ')) \
                  .filter(lambda x: len(x) > 0) \
                  .map(lambda x: (x, 1)) \
                  .reduceByKey(add)

    output = counts.collect()
    output.sort(key = lambda x: x[0], reverse = False)
    output.sort(key = lambda x: x[1], reverse = True)

    output_filename = os.path.expanduser('~/spark/local/word_count_1.out')
    f = open(output_filename, 'wt')
    for (word, count) in output:
        print("%s: %i" % (word, count), file=f)
    f.close()
    print()
    print(f'You can find the result of this word count in the file {output_filename}')
    print()

    spark.stop()
