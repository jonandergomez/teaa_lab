import os
import sys
import numpy
from matplotlib import pyplot

from sklearn.metrics import pairwise_distances


filename = 'models/cluster-distribution-066.csv'

if len(sys.argv) > 1:
    filename = sys.argv[1]

mat = numpy.genfromtxt(filename, delimiter = ';')

print(mat.shape)

for i in range(mat.shape[0]): mat[i, :] /= sum(mat[i, :])

conf_mat = pairwise_distances(mat)

fig, axis = pyplot.subplots(nrows = 1, ncols = 1, figsize = (8, 8))
#
axis.matshow(conf_mat, cmap = pyplot.cm.Greens, alpha = 1.0)
axis.set_xlabel('Target classes')
axis.set_ylabel('Target classes')
axis.set_title('Confusion matrix')
pyplot.tight_layout()
pyplot.show()
