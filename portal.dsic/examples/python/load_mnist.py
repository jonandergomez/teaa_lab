import os
import sys
import numpy
import pickle
from sklearn.datasets import fetch_openml

def load_mnist():
    #
    home = os.getenv('HOME')
    filename = None
    if home is not None:
        filename = home + '/teaa/examples/data/mnist_784.npz'
    else:
        filename = '/home/ubuntu/teaa/examples/mnist_784.npz' # This will fail in Windows machines
    #
    if os.path.exists(filename):
        npz = numpy.load(filename, allow_pickle = True)
        X, y = npz['X'], npz['y']
    else:
        # This fails from the master of the Spark cluster because it is not allowed to access internet
        X, y = fetch_openml('mnist_784', version = 'active', return_X_y = True, parser = 'auto')
        numpy.savez(filename, X = X, y = y)
    #
    y = numpy.array([int(_) for _ in y])
    return X, y
