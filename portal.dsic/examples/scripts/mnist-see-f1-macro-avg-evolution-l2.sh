#!/bin/bash 

dataset="digits"
results_dir="results/${dataset}/ensembles"
target_dir="results/summaries"

for subset in "train" "test"
do
    for technique in "rf" "ert" "gbt" "gbt.b" # "gbt.r"
    do

        grep -H " macro avg " ${results_dir}/${technique}/${subset}/*.txt \
            | sed 's/-/ /g' \
            | sed 's/_/ /g' \
            | sed 's/\.txt/ /g' \
            | awk '{ print $2, $4, $6, $(NF-1) }' \
            | awk '{ for(i=1; i <= NF; i++) printf("%s;", $i); printf("\n"); }' >/tmp/temp.${technique}.$$


        csv_file="/tmp/${dataset}-f1-macro-avg-evolution-${technique}-${subset}.csv"
        (echo "num trees;pca;max depth;f1 macro avg" ; cat /tmp/temp.${technique}.$$) >${csv_file}

        rm -f /tmp/temp.${technique}.$$

        mkdir -p ${target_dir}
        mv ${csv_file} ${target_dir}/
   done
done
