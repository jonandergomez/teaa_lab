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

    def fit(self, X, y, min_samples_to_split = None):
        if RDD is not None and (isinstance(X, RDD) or isinstance(y, RDD)):
            raise Exception('RDD objects not supported for training')

        self.balltree = BallTree(min_samples_to_split = min_samples_to_split)
        self.balltree.fit(X, y)
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
            probs = sample.map(lambda t: (t[0], self.compute_probs_one_sample(t[1])))
            return probs.map(lambda t: (t[0], t[1].argmax())).collect()

        else:
            raise Exception(f'Not accepted data type: {type(sample)}')
    # ------------------------------------------------------------------------------

    def compute_probs_one_sample(self, x):
        if self.balltree is not None:
            if len(x.shape) == 2:
                y = numpy.zeros(self.num_classes)
                for n in range(len(x)):
                    knn = self.balltree.get_knn(x[n], self.K)
                    if len(knn) > self.K: knn = knn[:self.K]
                    for t in knn: y[t[0]] += 1
            elif len(x.shape) == 1:
                knn = self.balltree.get_knn(x, self.K)
                if len(knn) > self.K: knn = knn[:self.K]
                y = numpy.zeros(self.num_classes)
                for t in knn: y[t[0]] += 1
            else:
                raise Exception(f'Unexpected shape of a sample {x.shape}')
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
            return sample.map(lambda t: (t[0], self.compute_probs_one_sample(t[1]))).collect()

        else:
            raise Exception(f'Not accepted data type: {type(sample)}')
    # ------------------------------------------------------------------------------
