#!/bin/bash


for p in {01..24}
do
    patient="chb${p}"
    for num_trees in 50 100 200 300 500 700 1000
    do
        for classification_type in "--doBinaryClassification" "--no-doBinaryClassification"
        do
            for using_pca in "--usingPCA" "--no-usingPCA"
            do
                scripts/run-python.sh python/rf_uc13.py ${patient} --numTrees ${num_trees} ${classification_type} ${using_pca} --verbose 0
            done
        done
    done
done
