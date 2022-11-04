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
            | awk '{ print $2, $4, $6, $12 }' \
            | awk '{ for(i=1; i <= NF; i++) printf("%s;", $i); printf("\n"); }' >/tmp/temp.${technique}.$$


        (echo "num_trees;pca;max_depth;macro avg" ; cat /tmp/temp.${technique}.$$) >/tmp/${dataset}-${technique}-${subset}-accuracy-evolution.csv

        rm -f /tmp/temp.${technique}.$$

#cat <<EOF
#    
#        See the file /tmp/${dataset}-${technique}-${subset}-accuracy-evolution.csv and use it to represent a plot 
#        of accuracy versus several configuration parameters of ${technique}
#EOF
        mkdir -p results.summary/l2/${dataset}
        mv /tmp/${dataset}-${technique}-${subset}-accuracy-evolution.csv results.summary/l2/${dataset}/
    done
done
