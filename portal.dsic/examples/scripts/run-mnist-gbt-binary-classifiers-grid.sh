#!/bin/bash

for pca in 0.95 # 37 41 53 71 0.95
do
    num_trees="10:20:30:50:100:200"
    max_depth="3:5:7:9:11"
    scripts/run-python.sh python/gbt_mnist_binary_trees_2023.py \
                                --numTrees ${num_trees} \
                                --maxDepth ${max_depth} \
                                --pcaComponents ${pca} \
                                --verbose 0
done
