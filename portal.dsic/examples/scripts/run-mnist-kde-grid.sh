#!/bin/bash


for pca in 37 41 53 67 0.95
do
    #kmeans_codebook_sizes="0:100:200:300"
    kmeans_codebook_sizes="0:300"
    band_width="0.1:0.2:0.5:1.0:2.0"

    scripts/run-python-2.sh python/kde_mnist.py \
                                --codebookSize ${kmeans_codebook_sizes} \
                                --bandWidth ${band_width} \
                                --pcaComponents ${pca} \
                                --verbose 0
done
