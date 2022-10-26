#!/bin/bash

for patient in {01..24}
do
    python3 python/do_pca_uc13_21x20.py --patient "chb${patient}"
done
