#!/bin/bash


#for pca in 37 41 53 67 0.95
for pca in 37
do
    kmeans_codebook_sizes="0:100:200:300:500"
    K="3:5:7:9:11:13"

    scripts/run-python-2.sh python/knn_mnist.py \
                                --codebookSize ${kmeans_codebook_sizes} \
                                --K ${K} \
                                --pcaComponents ${pca} \
                                --verbose 0
done
