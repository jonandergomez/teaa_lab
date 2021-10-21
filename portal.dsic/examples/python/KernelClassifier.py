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

try:
    from pyspark.rdd import RDD
except:
    RDD = None

class KernelClassifier:
    """
    This class implements a classifier based on Kernel Density Estimator.

    The purpose is to classify each sample according to the class with higher probability density.

    """
    
    def __init__(self, band_width = None):
        self.num_classes = None
        self.dim = None
        self.targets = None
        self.samples = None
        self.sizes = None
        self.band_width = band_width
        self.kernel = 'gaussian' # This could be a parameter for the constructor, but the
                                 # current implementation of MyKernel.py doesn't allow a
                                 # different kernel type.
    # ------------------------------------------------------------------------------

    def fit(self, X, y):
        if RDD is not None and (isinstance(X, RDD) or isinstance(y, RDD)):
            raise Exception('RDD objects not supported for training')

        # Get the labels
        self.targets = numpy.unique(y)
        self.num_classes = len(self.targets)
        # Get the sample dimension
        self.dim = X.shape[1]
        # Establish the value of 'band_width' if not set previously
        if self.band_width is None:
            self.band_width = math.sqrt(sum(numpy.ones(self.dim) ** 2))

        # Separate the training samples of each class in order to do the estimation
        self.samples = list()
        self.labels = list()
        self.sizes = list()
        for k in range(self.num_classes):
            self.samples.append(X[y == self.targets[k]])
            self.labels.append(y[y == self.targets[k]])
            self.sizes.append(len(self.samples[-1]))
        #
        return self
    # ------------------------------------------------------------------------------

    def kernel_for_probs(self, sample):
        if len(sample.shape) == 2:
            densities = list()
            for n in range(len(sample)):
                densities.append([numpy.exp(- 0.5 * ((sample[n] - self.samples[i]) ** 2).sum(axis = 1) / self.band_width ** 2).sum() / self.sizes[i] for i in range(self.num_classes)])
            densities = numpy.array(densities)
            densities = densities.sum(axis = 0)
        else:
            densities = [numpy.exp(- 0.5 * ((sample - self.samples[i]) ** 2).sum(axis = 1) / self.band_width ** 2).sum() / self.sizes[i] for i in range(self.num_classes)]
            densities = numpy.array(densities)
        probs = densities / max(1.0e-3, densities.sum())
        return probs
    # ------------------------------------------------------------------------------

    def predict_probs(self, sample):
        #
        if type(sample) == list or type(sample) == numpy.ndarray and len(sample.shape) >= 2:
            probs = list()

            for i in range(len(sample)):
                probs.append(self.kernel_for_probs(sample[i]))
                print("%10d" % i, end = '\r')

            return numpy.array(probs) # -> numpy.ndarray with shape (len(sample), num_classes)

        elif type(sample) == numpy.ndarray and len(sample.shape) == 1:
            return self.kernel_for_probs(sample) # -> numpy.ndarray with shape (num_classes,)

        elif RDD is not None and isinstance(sample, RDD):
            labels_and_probs = sample.map(lambda t: (t[0], self.kernel_for_probs(t[1]))).collect()
            # -> tuple( numpy.ndarray with shape (sample.count(),) ,  numpy.ndarray with shape (sample.count(), num_classes) )
            return numpy.array([t[0] for t in labels_and_probs]), numpy.array([t[1] for t in labels_and_probs])

        else:
            raise Exception(f'Not accepted data type: {type(sample)}')
    # ------------------------------------------------------------------------------

    def predict(self, sample):
        probs = self.predict_probs(sample)
        if type(probs) == tuple:
            return probs[0], probs[1].argmax(axis = 1)
        elif len(probs.shape) == 1:
            return probs.argmax()
        else:
            return probs.argmax(axis = 1)
    # ------------------------------------------------------------------------------
