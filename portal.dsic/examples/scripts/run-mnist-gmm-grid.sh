#!/bin/bash

convergence_tolerance="1.0e-3"
covar_type="full"

for J in 3 5 7 10 11 12 13 14 15 17 19 20 23 25 27 30
do
    for pca_components in 11 23 37 41 53 67
    do
        scripts/run-python.sh python/gmm_mnist_2022.py ${J} \
                                                        --pca ${pca_components} \
                                                        --convergenceTol ${convergence_tolerance} \
                                                        --covarType ${covar_type} \
                                                        --minVar 1.0
    done
done


convergence_tolerance="1.0e-4"
covar_type="diagonal"

for J in 10 15 20 30 40 50 70 100
do
    for pca_components in 11 23 37 41 53 67
    do
        scripts/run-python.sh python/gmm_mnist_2022.py ${J} \
                                                        --pca ${pca_components} \
                                                        --convergenceTol ${convergence_tolerance} \
                                                        --covarType ${covar_type} \
                                                        --minVar 1.0e-5
    done
done
