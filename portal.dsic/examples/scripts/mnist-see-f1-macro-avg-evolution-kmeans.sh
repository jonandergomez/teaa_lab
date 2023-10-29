#!/bin/bash 

dataset="digits"
results_dir="results/${dataset}/kmeans"
target_dir="results/summaries"

for subset in  "test" "train"
do
    grep -H "^   macro avg" ${results_dir}/${subset}/kmeans-classification-results*.txt \
        | sed 's/-/ /g' \
        | awk '{ print $5, $7, $9, $11, $(NF-1) }' \
        | sed 's/\.txt:/ /g' \
        | awk '{ for(i=1; i <= NF; i++) printf("%s;", $i); printf("\n"); }' >/tmp/temp.${subset}.$$


    csv_file="/tmp/${dataset}-f1-macro-avg-evolution-kmeans-${subset}.csv"

    (echo "J;pca;min var;covar type;f1 macro avg" ; cat /tmp/temp.${subset}.$$) >${csv_file}

    rm -f /tmp/temp.${subset}.$$

    mkdir -p ${target_dir}
    mv ${csv_file} ${target_dir}

    echo ""
    echo "Results can be found in the following directory:"
    echo ${target_dir}
    ls -l ${target_dir}
    echo ""
done
