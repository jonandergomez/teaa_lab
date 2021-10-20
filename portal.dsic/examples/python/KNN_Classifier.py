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

from BallTree import BallTree

class KNN_Classifier:
    """
    This class implements a classifier based on K-Nearest Neighbours.

    The purpose is to classify each sample according to the class with majority within the set of K-Nearest samples.

    """
    
    def __init__(self, K = 5, num_classes = 2):
        self.num_classes = num_classes
        self.K = K
        self.balltree = None
    # ------------------------------------------------------------------------------

    def unpersist(self):
        #self.balltree.unpersist()
    # ------------------------------------------------------------------------------

    def fit(self, Xy, min_samples_to_split = 100, max_bins = 32):
        #if not isinstance(Xy, RDD):
        #    raise Exception('An RDD object must be provided')

        self.balltree = BallTree(min_samples_to_split = min_samples_to_split)
        self.balltree.fit(Xy)
        #
        return self
    # ------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------
    def predict(self, sample):
        #
        if type(sample) == list or type(sample) == numpy.ndarray and len(sample.shape) >= 2:
            probs = self.predict_probs(sample)
            return probs.argmax(axis = 1)

        elif type(sample) == numpy.ndarray and len(sample.shape) == 1:
            probs = self.predict_probs(sample)
            return probs.argmax()

        elif RDD is not None and isinstance(sample, RDD):
            raise Exception('Not supported yet!')

        else:
            raise Exception(f'Not accepted data type: {type(sample)}')
    # ------------------------------------------------------------------------------

    def compute_probs_one_sample(self, x):
        if self.balltree is not None:
            knn = self.balltree.get_knn(x, self.K)
            if len(knn) > self.K: knn = knn[:self.K]
            y = numpy.zeros(self.num_classes)
            for t in knn: y[t[0]] += 1
            return y / (1.0e-6 + y.sum())

    def predict_probs(self, sample):
        #
        if type(sample) == list or type(sample) == numpy.ndarray and len(sample.shape) >= 2:
            densities = list()
            i = 1
            for x in sample:
                densities.append(self.compute_probs_one_sample(x))
                print("predicted %10d samples" % i, end = '\r')
                i += 1
            #
            return numpy.array(densities)

        elif type(sample) == numpy.ndarray and len(sample.shape) == 1:
            return self.compute_probs_one_sample(sample)

        elif RDD is not None and isinstance(sample, RDD):
            raise Exception('Not supported yet!')

        else:
            raise Exception(f'Not accepted data type: {type(sample)}')
    # ------------------------------------------------------------------------------
