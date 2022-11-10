#!/bin/bash 

dataset="mnist"

for subset in "train" "test"
do
    for technique in "rf" "ert" "gbt"
    do

        grep -H "^   macro avg" results.l2.${dataset}.${subset}/${technique}/${technique}_*.txt \
            | sed 's/-/ /g' \
            | sed 's/_/ /g' \
            | sed 's/\.txt/ /g' \
            | awk '{ print $2, $4, $6, $(NF-1) }' \
            | awk '{ for(i=1; i <= NF; i++) printf("%s;", $i); printf("\n"); }' >/tmp/temp.${technique}.$$


        (echo "num trees;pca;max depth;f1 macro avg" ; cat /tmp/temp.${technique}.$$) >/tmp/${dataset}-${technique}-${subset}-f1-macro-avg-evolution.csv

        rm -f /tmp/temp.${technique}.$$

        mkdir -p results.summary/l2/${dataset}
        mv /tmp/${dataset}-${technique}-${subset}-f1-macro-avg-evolution.csv results.summary/l2/${dataset}/
   done
done
