import os
import sys
import time
import pickle
import math
import numpy
import tempfile

use_spark = True
try:
    from pyspark import SparkContext
except:
    use_spark = False


# --------------------------------------------------------------------------------
def generate_y_true(time_to_seizure):
    tts = time_to_seizure
    y_true = numpy.zeros(len(tts), dtype = int)
    y_true[tts >   2     ] = 1
    y_true[tts >  10 * 60] = 2
    y_true[tts >  20 * 60] = 3
    y_true[tts <    -10  ] = 4
    y_true[tts < -20 * 60] = 5
    y_true[tts == 0      ] = 0 # redundant, but just in case
    return y_true
# --------------------------------------------------------------------------------
def csv_line_to_patient_tts_label_and_sample_binary_classification(line):
    parts = line.split(';')
    patient = parts[0]
    tts = float(parts[1])
    label = 0 if tts == 0 else 1
    x = numpy.array([float(x) for x in parts[2:]])
    return (patient, tts, label, x)
# --------------------------------------------------------------------------------
def csv_line_to_patient_tts_label_and_sample_multiclass_classification(line):
    parts = line.split(';')
    patient = parts[0]
    tts = float(parts[1])
    label = 0
    ### BEGIN: Students can change this to check other approaches
    if tts >   2     : label = 1
    if tts >  10 * 60: label = 2
    if tts >  20 * 60: label = 3
    if tts < -10     : label = 4
    if tts < -20 * 60: label = 5
    ### END: Students can change this to check other approaches
    x = numpy.array([float(x) for x in parts[2:]])
    return (patient, tts, label, x)
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
def load_csv_from_uc13(spark, filenames, num_partitions = None, do_binary_classification = False):
    if spark is None:
        data = []
        for filename in filenames:
            print(f'loading {filename}')
            f = gzip.open(filename, 'rt')
            for line in f:
                if do_binary_classification:
                    data.append(csv_line_to_patient_tts_label_and_sample_binary_classification(line.strip()))
                else:
                    data.append(csv_line_to_patient_tts_label_and_sample_multiclass_classification(line.strip()))
            f.close()
        print(f'loaded {len(data)} samples')
        return data
    else:
        rdd = None
        for filename in filenames:
            print(f'loading {filename}')
            csv_lines = spark.textFile(filename)
            if num_partitions is not None:
                csv_lines = csv_lines.repartition(num_partitions)
            if do_binary_classification:
                csv_lines = csv_lines.map(csv_line_to_patient_tts_label_and_sample_binary_classification)
            else:
                csv_lines = csv_lines.map(csv_line_to_patient_tts_label_and_sample_multiclass_classification)
            if rdd is not None:
                rdd = rdd.union(csv_lines)
            else:
                rdd = csv_lines
        print(f'loaded {rdd.count()} samples into {rdd.getNumPartitions()} partitions')
        return rdd
# --------------------------------------------------------------------------------
