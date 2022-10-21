#!/bin/bash 

grep -H "^weighted avg" results.l1.digits.2022.test/gmm-classification-results*.txt \
    | sed 's/-/ /g' \
    | awk '{ print $5, $7, $9, $11, $14 }' \
    | sed 's/\.txt:weighted//g' \
    | awk '{ for(i=1; i <= NF; i++) printf("%s;", $i); printf("\n"); }' >/tmp/temp.test.$$


(echo "J;pca;min_var;covar_type;accuracy" ; cat /tmp/temp.test.$$) >/tmp/mnist-gmm-accuracy-evolution.csv

rm -f /tmp/temp.test.$$


cat <<EOF
    
    See the file /tmp/mnist-gmm-accuracy-evolution.csv or use it to represent a plot 
    of accuracy versus several configuration variables obtained by using GMMs

EOF

