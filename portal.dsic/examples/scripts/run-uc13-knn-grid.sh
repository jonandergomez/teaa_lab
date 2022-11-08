#!/bin/bash

patient_list="03 07 08 10 12 13 15 16 22 24"
kmeans_codebook_sizes="0:100:200:300"
K="3:5:7:9:11:13"

for p in ${patient_list}
do
    patient="chb${p}"

    for do_binary_classification in "--doBinaryClassification" "--no-doBinaryClassification"
    do
        for using_pca in "--usingPCA" "--no-usingPCA"
        do
            scripts/run-python-2.sh python/knn_uc13_21x20.py ${patient} \
                                        --codebookSize ${kmeans_codebook_sizes} \
                                        --K ${K} \
                                        ${using_pca} ${do_binary_classification} \
                                        --verbose 0
        done
    done
done
