import os
import sys
import numpy

home = os.getenv("HOME")
sys.path.append(home + '/machine_learning_for_students')

from machine_learning.generate_datasets import generate_multivariate_normals

N = 1000 * 1000 # number of samples
D = 30 # dimension
K = 10 # number of different clusters


X_train, Y_train, X_test, Y_test = generate_multivariate_normals(n_classes = K,
                                                                 dimensionality = D,
                                                                 n_samples_per_class_training = N // K,
                                                                 n_samples_per_class_test = N // K // 5,
                                                                 sparsity_between_classes = 1.0,
                                                                 sparsity_within_a_class = 1.0)


print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)


def to_csv_line(x):
    return ';'.join('{:.4f}'.format(_) for _ in x)

f = open(f'data/vectors_train_{len(X_train)}_{D}_{K}.csv', 'wt')
for x in X_train:
    f.write(to_csv_line(x))
    f.write('\n')
f.close()
#
f = open(f'data/labels_train_{len(Y_train)}_{K}.csv', 'wt')
for y in Y_train:
    f.write('%d\n' % y)
f.close()
#
f = open(f'data/vectors_test_{len(X_test)}_{D}_{K}.csv', 'wt')
for x in X_test:
    f.write(to_csv_line(x))
    f.write('\n')
f.close()
#
f = open(f'data/labels_test_{len(Y_test)}_{K}.csv', 'wt')
for y in Y_test:
    f.write('%d\n' % y)
f.close()
