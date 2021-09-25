import os
import sys
import numpy
from matplotlib import pyplot


filename = 'models/cluster-distribution-066.csv'

if len(sys.argv) > 1:
    filename = sys.argv[1]

mat = numpy.genfromtxt(filename, delimiter = ';')

print(mat.shape)

for i in range(mat.shape[0]): mat[i, :] /= sum(mat[i, :])

height_inches = 4
width_inches = min(max(height_inches, mat.shape[1] * 0.4), 30)

fig, axis = pyplot.subplots(nrows = 1, ncols = 1, figsize = (width_inches, height_inches))
#
axis.matshow(mat, cmap = pyplot.cm.Blues, alpha = 1.0)
axis.set_xlabel('Clusters')
axis.set_ylabel('Target classes')
axis.set_title('Conditional probabilities')
pyplot.tight_layout()
pyplot.show()
