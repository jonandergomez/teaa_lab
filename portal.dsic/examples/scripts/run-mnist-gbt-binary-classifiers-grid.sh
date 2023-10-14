#!/bin/bash

for pca in 0.95 # 37 41 53 71 0.95
do
    #num_trees="10:15:20:25:30:50:100:200"
    num_trees="200:300:500"
    #max_depth="3:5:7:9:11" #:13"
    max_depth="3:5:7"
    scripts/run-python.sh python/gbt_mnist_binary_trees.py \
                                --numTrees ${num_trees} \
                                --maxDepth ${max_depth} \
                                --pcaComponents ${pca} \
                                --verbose 0
done
