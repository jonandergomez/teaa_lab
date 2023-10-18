#!/bin/bash

# patient 11 must be removed
for p in 03 07 08 09 10 11 12 13 15 16 18 19 22 24
do
    patient="chb${p}"
    num_trees="20:30:50:100:200"
    #num_trees="100"
    max_depth="3:5:7:9"
    #max_depth="5"
    format="pca136"
    #format="21x14"

    scripts/run-python.sh python/gbt_uc13_regression.py ${patient} \
                                                            --numTrees ${num_trees} \
                                                            --maxDepth ${max_depth} \
                                                            --dataFormat ${format} \
                                                            --verbose 0
done
