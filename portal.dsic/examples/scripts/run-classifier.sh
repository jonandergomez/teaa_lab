#!/bin/bash

case $(hostname) in
    teaa-*-cluster*)
        num_partitions=40
        ;;
    *)
        num_partitions=80
        ;;
esac

for q in $(ls models/kmeans_model-uc13* | cut -f3 -d'-' | cut -f1 -d'.')
do
    codebook=$(printf "models/kmeans_model-uc13-${q}.pkl")
    if [ -f ${codebook} ]
    then
        #echo "we have this codebook ${codebook}"
        counter_pairs=$(printf "models/cluster-distribution-${q}.csv")
        if [ -f ${counter_pairs} ]
        then
            results_file=$(printf "results3/classification-results-${q}.txt")
            if [ ! -f ${results_file} ]
            then
                scripts/run-python.sh python/kmeans_uc13_classifier.py  \
                                        --num-partitions ${num_partitions} \
                                        --codebook ${codebook} \
                                        --counter-pairs ${counter_pairs}
            fi
        fi
    fi
done
