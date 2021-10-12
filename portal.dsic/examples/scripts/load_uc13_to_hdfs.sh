#!/bin/bash

source_dir="/bigdata/disk/jon/deephealth/UC13"
hdfs_dest_dir="data/uc13"

for i in {1..24}
do
    patient=$(printf "chb%02d" ${i})

    for subset in train test
    do
        filename="uc13-${patient}-21x20-${subset}.csv"
        cat ${source_dir}/${filename} | hdfs dfs -put - ${hdfs_dest_dir}/${filename}
    done
done
