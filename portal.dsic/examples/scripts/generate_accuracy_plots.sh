#!/bin/bash


python python/see_accuracy_evolution.py  --results-dir results  --kmeans --classification
python python/see_accuracy_evolution.py  --results-dir results2 --kmeans --classification


python python/see_accuracy_evolution.py  --results-dir results3.train --gmm --classification
python python/see_accuracy_evolution.py  --results-dir results3.test  --gmm --classification

python python/see_accuracy_evolution.py  --results-dir results4.pca.train --kmeans --classification
python python/see_accuracy_evolution.py  --results-dir results4.pca.train --gmm    --classification
python python/see_accuracy_evolution.py  --results-dir results4.pca.train --gmm    --prediction
python python/see_accuracy_evolution.py  --results-dir results4.pca.test  --gmm    --classification

