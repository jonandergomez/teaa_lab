#!/bin/bash


for pca in 0.95 # 37 41 53 71 0.95
do
    num_trees="10:20:30:50:100:200:300:500:700:1000"
    max_depth="3:5:7:9:11"
    scripts/run-python.sh python/rf_mnist.py \
                                --numTrees ${num_trees} \
                                --maxDepth ${max_depth} \
                                --pcaComponents ${pca} \
                                --verbose 0
done
