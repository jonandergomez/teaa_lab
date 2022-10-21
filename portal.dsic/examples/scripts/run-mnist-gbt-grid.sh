#!/bin/bash


for pca in 41 53 0.95 # 37
do
    #num_trees="10:20:30:50:100:200:300:500:700:1000"
    num_trees="10:15:20:25:30:50:100"
    max_depth="3:5:7:9:11:13"
    scripts/run-python.sh python/gbt_mnist.py \
                                --numTrees ${num_trees} \
                                --maxDepth ${max_depth} \
                                --pcaComponents ${pca} \
                                --verbose 0
done
