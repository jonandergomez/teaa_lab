import os
import sys
import numpy
from matplotlib import pyplot

from sklearn.metrics import pairwise_distances

filename = None
do_save_fig = False
results_dir = None
clustering_type = 'kmeans'

for i in range(len(sys.argv)):
    if sys.argv[i] == '--filename':
        filename = sys.argv[i + 1]
    elif sys.argv[i] == '--kmeans':
        clustering_type = 'kmeans'
    elif sys.argv[i] == '--gmm':
        clustering_type = 'gmm'
    elif sys.argv[i] == '--results-dir':
        results_dir = sys.argv[i + 1]
        do_save_fig = True
    elif sys.argv[i] == '--save-figs':
        do_save_fig = True


mat = numpy.genfromtxt(filename, delimiter = ';')

#print(mat.shape)

for i in range(mat.shape[0]): mat[i, :] /= sum(mat[i, :])

conf_mat = pairwise_distances(mat)

fig, axis = pyplot.subplots(nrows = 1, ncols = 1, figsize = (8, 8))
#
axis.matshow(conf_mat, cmap = pyplot.cm.Greens, alpha = 1.0)
axis.set_xlabel('Target classes')
axis.set_ylabel('Target classes')
axis.set_title('Confusion matrix')
pyplot.tight_layout()
if do_save_fig and results_dir is not None:
    pyplot.savefig(f'{results_dir}/confusion-matrix-{clustering_type}-%d-%04d.svg' % (mat.shape[0], mat.shape[1]), format = 'svg')
    pyplot.savefig(f'{results_dir}/confusion-matrix-{clustering_type}-%d-%04d.png' % (mat.shape[0], mat.shape[1]), format = 'png')
else:
    pyplot.show()
