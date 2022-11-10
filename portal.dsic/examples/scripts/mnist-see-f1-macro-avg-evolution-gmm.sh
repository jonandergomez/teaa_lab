#!/bin/bash 

dataset="mnist"

for subset in  "test"  # "train"
do

    grep -H "^   macro avg" results.l1.digits.2022.${subset}/gmm-classification-results*.txt \
        | sed 's/-/ /g' \
        | awk '{ print $5, $7, $9, $11, $(NF-1) }' \
        | sed 's/\.txt:/ /g' \
        | awk '{ for(i=1; i <= NF; i++) printf("%s;", $i); printf("\n"); }' >/tmp/temp.${subset}.$$


    csv_file="/tmp/${dataset}-gmm-f1-macro-avg-evolution-${subset}.csv"
    target_dir="results.summary/l1/${dataset}/gmm"

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
