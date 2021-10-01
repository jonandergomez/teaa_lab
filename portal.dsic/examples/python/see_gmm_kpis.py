import os
import sys
import numpy
from matplotlib import pyplot


filename = 'log/OUT'

if len(sys.argv) > 1:
    filename = sys.argv[1]

n_components = list()
log_likelihood = list()
aic = list()
bic = list()

f = open(filename,'rt')
for line in f:
    parts = line.split()
    if parts[0] == 'n_components':
        # n_components     1  logL = -1.010794e+04  aic 9.444660e+08  bic 9.444664e+08 
        n_components.append(int(parts[1]))
        log_likelihood.append(float(parts[4]))
        aic.append(float(parts[6]))
        bic.append(float(parts[8]))
f.close()


fig, axes = pyplot.subplots(nrows = 1, ncols = 1, figsize = (9, 6))
#
axis = axes
axis.set_facecolor('#eefffc')
axis.plot(n_components, aic, 'r-', alpha = 1.0, label = 'AIC')
axis.plot(n_components, bic, 'b-', alpha = 1.0, label = 'BIC')
axis.grid()
axis.set_xlabel('Number of clusters')
axis.set_ylabel('AIC & BIC criteria')
axis.set_title('Evolution of AIC & BIC as the number of Gaussian components increases')
axis.legend()
#
pyplot.tight_layout()
pyplot.show()
