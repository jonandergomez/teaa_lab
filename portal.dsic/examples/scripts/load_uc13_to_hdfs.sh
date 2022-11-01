#!/bin/bash

#source_dir="/bigdata/disk/jon/deephealth/uc13/21x20"
source_dir="${HOME}/data"
hdfs_dest_dir="data/uc13"

for i in {1..24}
do
    patient=$(printf "chb%02d" ${i})

    for subset in train test
    do
        #filename="uc13-${patient}-21x20-${subset}.csv"
        #filename="uc13-${patient}-21x20-${subset}-pca.csv"
        #filename="uc13-${patient}-21x20-time-to-seizure-${subset}.csv"
        filename="uc13-${patient}-21x20-time-to-seizure-${subset}-pca.csv"
        echo ${filename}
        cat ${source_dir}/${filename} | hdfs dfs -put - ${hdfs_dest_dir}/${filename}
    done
done
