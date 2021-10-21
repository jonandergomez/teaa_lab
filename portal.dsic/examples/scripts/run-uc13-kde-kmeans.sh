#!/bin/bash

for n in {1..24}
do
    patient=$(printf "chb%02d" ${n})

    scripts/run-python-2.sh python/kde_uc13_21x20.py \
            --patient ${patient} \
            --reduce-labels \
            --band-width 2.0 >${HOME}/uc13-21x20/logs/out.${patient} 2>${HOME}/uc13-21x20/logs/err.${patient}
done
