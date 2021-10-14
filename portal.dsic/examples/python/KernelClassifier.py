"""
    Author: Jon Ander Gomez Adrian (jon@dsic.upv.es, http://personales.upv.es/jon)
    Version: 1.0
    Date: October 2021
    Universitat Politecnica de Valencia
    Technical University of Valencia TU.VLC

"""

import sys
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
    def __del__(self):
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
        bd = self.band_width
        if type(sample) == list or type(sample) == numpy.ndarray and len(sample.shape) >= 2:
            densities = dict()
            if type(sample) == list:
                for label in self.targets:
                    densities[label] = self.samples[label] \
                            .map(lambda x: kernel_for_lists(x, sample, bd)) \
                            .reduce(lambda x, y: x + y) / self.sizes[label]
            else:
                for label in self.targets:
                    densities[label] = self.samples[label] \
                            .map(lambda x: kernel_for_numpy(x, sample, bd)) \
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
                            .map(lambda x: kernel_for_one_sample(x, sample, bd)) \
                            .reduce(lambda x, y: x + y) / self.sizes[label]
                if density > max_density:
                    max_density = density
                    predicted_label = label
            return predicted_label

        else:
            raise Exception(f'Not accepted data type: {type(sample)}')
    # ------------------------------------------------------------------------------
