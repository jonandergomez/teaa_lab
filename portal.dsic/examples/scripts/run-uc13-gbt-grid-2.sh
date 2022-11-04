#!/bin/bash

# patient 11 must be removed
for p in 03 07 08 10 11 12 13 15 16 18 19 22 24
do
    patient="chb${p}"
    num_trees="20:30:50"
    max_depth="5:7:9:11"
    for using_pca in "--usingPCA" "--no-usingPCA"
    do
        scripts/run-python.sh python/gbt_uc13_binary_trees.py ${patient} \
                                                            --numTrees ${num_trees} \
                                                            --maxDepth ${max_depth} \
                                                            ${using_pca} \
                                                            --verbose 0
    done
done
