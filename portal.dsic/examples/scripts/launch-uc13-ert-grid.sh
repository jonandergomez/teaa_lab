#!/bin/bash

patient_list="03 07 08 10 11 12 13 15 16 18 19 22 24"

for p in ${patient_list}
do
    scripts/run-uc13-ert-grid.sh ${p} >${p}.out 2>${p}.err &
    echo "launched ${p}: see ${p}.out and ${p}.err"
    ls -l ${p}.out ${p}.err
done
