#!/bin/bash

patient_list="03 07 08 10 11 12 13 15 16 18 19 22 24"

if [ $# -ge 1 ]
then
    patient_list="$1"
fi

for p in ${patient_list}
do
    patient="chb${p}"
    for num_trees in 50 100 200 300 500 700 1000
    do
        for max_depth in 3 5 7 9 11 13
        do
            for classification_type in "--doBinaryClassification" "--no-doBinaryClassification"
            do
                for using_pca in "--usingPCA" "--no-usingPCA"
                do
                    python3 python/ert_uc13.py ${patient} \
                                            --numTrees ${num_trees} \
                                            --maxDepth ${max_depth} \
                                            ${classification_type} \
                                            ${using_pca} \
                                            --verbose 0
                done
            done
        done
    done
done
