#!/bin/bash

for p in 03 07 08 10 11 12 13 15 16 18 19 22 24
do
    patient="chb${p}"
    num_trees="10:15:20:25:30:50"
    max_depth="3:5:7:9:11:13"
    for classification_type in "--doBinaryClassification" # "--no-doBinaryClassification"
    do
        for using_pca in "--usingPCA" "--no-usingPCA"
        do
            scripts/run-python.sh python/gbt_uc13.py ${patient} \
                                                    --numTrees ${num_trees} \
                                                    --maxDepth ${max_depth} \
                                                    ${classification_type} \
                                                    ${using_pca} \
                                                    --verbose 0
        done
    done
done
