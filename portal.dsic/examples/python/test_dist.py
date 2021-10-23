import numpy
from sklearn.metrics.pairwise import euclidean_distances

x = numpy.random.randn(10, 11)
y = numpy.random.randn(100, 11)

xx = (x ** 2).sum(axis = 1)
yy = (y ** 2).sum(axis = 1)
xy = numpy.dot(x, y.T)

print(x.shape, xx.shape, xy.shape)
print(y.shape, yy.shape, xy.shape)


d1 = xx.reshape(-1, 1) - 2 * xy
print(d1.shape)
d1 = d1 + yy.reshape(1, -1)
print(d1.shape)

d2 = euclidean_distances(x, y, squared = True)
print(d2.shape)

print(abs(d1 - d2).max())
