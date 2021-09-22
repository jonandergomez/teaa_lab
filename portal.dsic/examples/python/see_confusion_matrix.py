import os
import sys
import numpy
from matplotlib import pyplot


mat = numpy.genfromtxt('results/classification-results-200.txt')

print(mat.shape)

mat1 = mat.copy()
mat2 = mat.copy()

for i in range(mat1.shape[0]): mat1[i, :] /= sum(mat1[i, :])
for j in range(mat2.shape[1]): mat2[:, j] /= sum(mat2[:, j])

fig, axes = pyplot.subplots(nrows = 1, ncols = 2, figsize = (11, 7))
axis = axes[0]
axis.matshow(mat1, cmap = pyplot.cm.Blues, alpha = 1.0)
axis.set_xlabel('Predictions')
axis.set_ylabel('True labels')
axis.set_title('Confusion Matrix')
axis = axes[1]
axis.matshow(mat2, cmap = pyplot.cm.Reds, alpha = 1.0)
axis.set_xlabel('Predictions')
axis.set_ylabel('True labels')
axis.set_title('Confusion Matrix')
pyplot.show()
