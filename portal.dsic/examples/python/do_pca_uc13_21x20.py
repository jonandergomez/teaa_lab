#
#       14009 "Scalable Machine Learning Techniques"
#
#   Bachelor's degree in Data Science
#   School of Informatics  (http://www.etsinf.upv.es)
#   Technical University of Valencia (http://www.upv.es)
#
#
import os
import sys
import time
import pickle
import math
import numpy

from sklearn.decomposition import PCA


if __name__ == "__main__":

    debug = 0
    num_channels = 21
    base_dir = 'uc13-21x20'
    patient = None

    for i in range(1, len(sys.argv)):
        if   sys.argv[i] == '--patient'                 : patient = sys.argv[i + 1]


    def csv_line_to_patient_label_and_sample(line):
        parts = line.split(';')
        patient = parts[0]
        label = int(parts[1])
        x = numpy.array([float(x) for x in parts[2:]])
        #x = x.reshape(21, -1)
        #x = x - x.min(axis = 1).reshape(-1, 1)
        #x = x / x.max(axis = 1).reshape(-1, 1)
        return (patient, label, x.flatten())

    data_rdds = dict()
    for subset in ['train', 'test']:
        #f = open(f'../21x20/uc13-{patient}-21x20-{subset}.csv')
        f = open(f'../21x20/uc13-{patient}-21x20-time-to-seizure-{subset}.csv')
        data = list()
        for line in f:
            data.append(csv_line_to_patient_label_and_sample(line.strip()))
        f.close()
        data_rdds[subset] = data
        num_samples = len(data)
        print(f'loaded {num_samples} {subset} samples')


    X_train = numpy.vstack([x[2] for x in data_rdds['train']])
    X_test  = numpy.vstack([x[2] for x in data_rdds['test']])

    print(X_train.shape, X_test.shape)

    pca = PCA(n_components = 0.95)
    pca.fit(X_train)
    x_train = pca.transform(X_train)
    x_test = pca.transform(X_test)
    print(x_train.shape, x_test.shape)

    if x_train.shape[1] > 50:
        pca = PCA(n_components = 50)
        pca.fit(X_train)
        x_train = pca.transform(X_train)
        x_test = pca.transform(X_test)
        print(x_train.shape, x_test.shape)

    X = dict()
    X['train'] = x_train
    X['test']  = x_test

    
    for subset in ['train', 'test']:
        rdd = data_rdds[subset]
        x = X[subset]
        #f = open(f'../21x20/uc13-{patient}-21x20-{subset}-pca.csv', 'wt')
        f = open(f'../21x20/uc13-{patient}-21x20-time-to-seizure-{subset}-pca.csv', 'wt')
        for i in range(len(rdd)):
            patient = rdd[i][0]        
            label = rdd[i][1]
            s1 = f'{patient};{label}'
            s2 = ';'.join("{:f}".format(v) for v in x[i])
            f.write(f'{s1};{s2}\n')
        f.close()
