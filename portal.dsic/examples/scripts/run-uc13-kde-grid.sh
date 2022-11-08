#!/bin/bash

patient_list="03 07 08 10 12 13 15 16 22 24"
kmeans_codebook_sizes="0:100:200:300"
band_width="0.1:0.2:0.5:1.0:2.0"

for p in ${patient_list}
do
    patient="chb${p}"

    for do_binary_classification in "--doBinaryClassification" "--no-doBinaryClassification"
    do
        for using_pca in "--usingPCA" "--no-usingPCA"
        do
            scripts/run-python-2.sh python/kde_uc13_21x20.py ${patient} \
                                        --codebookSize ${kmeans_codebook_sizes} \
                                        --bandWidth ${band_width} \
                                        ${using_pca} ${do_binary_classification} \
                                        --verbose 0
        done
    done
done
