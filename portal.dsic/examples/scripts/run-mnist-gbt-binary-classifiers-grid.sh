#!/bin/bash

scripts/run-python.sh python/gbt_mnist_binary_trees.py \
                                --numTrees "30" \
                                --maxDepth "7:9:11:13" \
                                --pcaComponents 53 \
                                --verbose 0

for pca in 29 37 41 53
do
    num_trees="10:15:20:25:30"
    max_depth="3:5:7:9:11:13"
    echo scripts/run-python.sh python/gbt_mnist_binary_trees.py \
                                --numTrees ${num_trees} \
                                --maxDepth ${max_depth} \
                                --pcaComponents ${pca} \
                                --verbose 0
done
