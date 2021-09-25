import os
import sys
import numpy
from matplotlib import pyplot


filename = 'log/kmeans-kpis.txt'

if len(sys.argv) > 1:
    filename = sys.argv[1]

mat = numpy.genfromtxt(filename)

print(mat.shape)

fig, axes = pyplot.subplots(nrows = 3, ncols = 1, figsize = (9, 9))
#
axis = axes[0]
axis.set_facecolor('#eefffc')
axis.plot(mat[:,0], mat[:, 1], 'ro--', alpha = 1.0)
axis.grid()
axis.set_xlabel('Number of clusters')
axis.set_ylabel('WSSSE')
axis.set_title('Within Set Sum of Square Error')
#
axis = axes[1]
axis.set_facecolor('#fffcee')
axis.plot(mat[:,0], mat[:, 2], 'ro--', alpha = 1.0)
axis.grid()
axis.set_xlabel('Number of clusters')
axis.set_ylabel('CH index')
axis.set_title('Calinski Harabasz index')
#
axis = axes[2]
axis.set_facecolor('#eefcff')
axis.plot(mat[:,0], mat[:, 3], 'ro--', alpha = 1.0)
axis.grid()
axis.set_xlabel('Number of clusters')
axis.set_ylabel('DB index')
axis.set_title('Davies Bouldin index')
#
pyplot.tight_layout()
pyplot.show()
