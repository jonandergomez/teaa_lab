"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://personales.upv.es/jon)
    Version: 1.0
    Date: October 2021
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

"""

import sys
import time
import math
import numpy

#from pyspark import SparkContext
from pyspark.rdd import RDD

def kernel_for_numpy(x, sample, band_width):
    return numpy.exp( - 0.5 * (((sample - x) / band_width) ** 2).sum(axis = 1))

def kernel_for_lists(x, sample, band_width):
    return numpy.array([math.exp( - 0.5 * (((z - x) / band_width) ** 2).sum()) for z in sample])

def kernel_for_one_sample(x, sample, band_width):
    return math.exp( - 0.5 * (((sample - x) / band_width) ** 2).sum())

def kernel_for_cartesian_rdd(pair, band_width):
    a = pair[0] # tuple: id, d-dimensional array
    x = pair[1] # should be a d-dimensional array
    xa = a[1] # d-dimensional array
    density = math.exp(-0.5 * (((xa -x) / band_width) ** 2).sum()) # one float
    return (a[0], density)

class KernelClassifier:
    """
    This class implements a classifier based on Kernel Density Estimator.

    The purpose is to classify each sample according to the class with higher probability density.

    """
    
    def __init__(self, spark_context, band_width = None):
        #self.spark_context = spark_context
        self.num_classes = 0
        self.dim = 0
        self.targets = None
        self.samples = None
        self.sizes = None
        self.band_width = band_width
        self.kernel = 'gaussian' # This could be a parameter for the constructor, but the
                                 # current implementation of MyKernel.py doesn't allow a
                                 # different kernel type.
    # ------------------------------------------------------------------------------
    def unpersist(self):
        for label, rdd in self.samples.items():
            rdd.unpersist()

    # ------------------------------------------------------------------------------
    def fit(self, Xy):
        if not isinstance(Xy, RDD):
            raise Exception('An RDD object must be provided')

        # Get the labels
        self.targets = numpy.unique(Xy.map(lambda sample: sample[0]).collect())
        self.num_classes = len(self.targets)
        # Get the sample dimension
        self.dim = Xy.first()[1].shape[0]
        # Establish the value of 'band_width' if not set previously
        if self.band_width is None:
            self.band_width = math.sqrt(sum(numpy.ones(self.dim) ** 2))

        # Separate the training samples of each class in order to do the estimation
        self.samples = dict()
        self.sizes = dict()
        for label in self.targets:
            self.samples[label] = Xy.filter(lambda sample: sample[0] == label).map(lambda sample: sample[1])
            self.samples[label].persist()
            self.sizes[label] = self.samples[label].count()
        #
        return self
    # ------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------
    def predict(self, sample):
        #
        bw = self.band_width
        if type(sample) == list or type(sample) == numpy.ndarray and len(sample.shape) >= 2:
            densities = dict()
            if type(sample) == list:
                for label in self.targets:
                    densities[label] = self.samples[label] \
                            .map(lambda x: kernel_for_lists(x, sample, bw)) \
                            .reduce(lambda x, y: x + y) / self.sizes[label]
            else:
                for label in self.targets:
                    densities[label] = self.samples[label] \
                            .map(lambda x: kernel_for_numpy(x, sample, bw)) \
                            .reduce(lambda x, y: x + y) / self.sizes[label]

            predicted_labels = list()
            for j in range(len(sample)):
                max_density = 0.0
                predicted_label = None
                for label in self.targets:
                    if densities[label][j] > max_density:
                        max_density = densities[label][j]
                        predicted_label = label
                predicted_labels.append(predicted_label)
            return predicted_labels

        elif type(sample) == numpy.ndarray and len(sample.shape) == 1:
            max_density = 0.0
            predicted_label = None
            for label in self.targets:
                density = self.samples[label] \
                            .map(lambda x: kernel_for_one_sample(x, sample, bw)) \
                            .reduce(lambda x, y: x + y) / self.sizes[label]
                if density > max_density:
                    max_density = density
                    predicted_label = label
            return predicted_label

        elif isinstance(sample, RDD):
            
            y_true, y_pred = self.predict_probs(sample)
            y_pred = y_pred.map(lambda t: (t[0], t[1].argmax()))
            return y_true.join(y_pred).map(lambda t: (t[1][0], t[1][1])) # the index (t[0]) is ignored

        else:
            raise Exception(f'Not accepted data type: {type(sample)}')
    # ------------------------------------------------------------------------------
    def predict_probs(self, sample):
        #
        bw = self.band_width
        if type(sample) == list or type(sample) == numpy.ndarray and len(sample.shape) >= 2:
            densities = list()
            if type(sample) == list:
                for label in range(len(self.targets)):
                    densities.append(self.samples[label] \
                            .map(lambda x: kernel_for_lists(x, sample, bw)) \
                            .reduce(lambda x, y: x + y) / self.sizes[label])
            else:
                for label in range(len(self.targets)):
                    densities.append(self.samples[label] \
                            .map(lambda x: kernel_for_numpy(x, sample, bw)) \
                            .reduce(lambda x, y: x + y) / self.sizes[label])

            densities = numpy.array(densities).T
            return densities / densities.sum(axis = 1).reshape(-1, 1)

        elif type(sample) == numpy.ndarray and len(sample.shape) == 1:
            max_density = 0.0
            predicted_label = None
            densities = list()
            for label in range(len(self.targets)):
                densities.append(self.samples[label] \
                            .map(lambda x: kernel_for_one_sample(x, sample, bw)) \
                            .reduce(lambda x, y: x + y) / self.sizes[label])
            densities = numpy.array(densities)
            return densities / densities.sum()

        elif isinstance(sample, RDD):

            num_partitions = sample.getNumPartitions()
            temp = sample.zipWithIndex()
            y_true = temp.map(lambda t: (t[1], t[0][0]))
            data_with_ids = temp.map(lambda t: (t[1], t[0][1]))

            densities = None
            for label in self.targets:
                last_time = time.time()
                #
                denominator = self.sizes[label]
                _ = data_with_ids.cartesian(self.samples[label]) \
                    .map(lambda pair: kernel_for_cartesian_rdd(pair, bw)) \
                    .reduceByKey(lambda a, b: a + b, num_partitions) \
                    .map(lambda t: (t[0], t[1] / denominator))

                if densities is None:
                    densities = _.map(lambda t: (t[0], [t[1]]))
                else:
                    densities = densities.join(_).map(lambda t: (t[0], t[1][0] + [t[1][1]]), num_partitions)
                #
                print('preparation of the computation of densities for label', label, 'in', time.time() - last_time, 'seconds')
            #
            def normalize(x):
                x = numpy.array(x)
                return x / x.sum()
            # two RDD objects are returned, first one with true labels, second one with probs, both with sample index
            return y_true, densities.map(lambda t: (t[0], normalize(t[1])))

        else:
            raise Exception(f'Not accepted data type: {type(sample)}')
    # ------------------------------------------------------------------------------
