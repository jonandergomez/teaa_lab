import os
import sys
import numpy

from matplotlib import pyplot

from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# ---------------------------------------------------------------------------------------------------
class MyArgmaxForPredictedLabels(BaseEstimator):
    def __init__(self, threshold = 0.5, classes_ = None):
        self.threshold = threshold
        self.classes_ = classes_
        self._estimator_type = 'classifier'

    def fit(self, X, y):
        raise Exception('No fit implemented in this class')
        return self

    def predict(self, y_probs):
        assert type(y_probs) == numpy.ndarray
        assert len(y_probs.shape) == 1
        return y_probs
# ---------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------
def save_results(results_dir, filename_prefix, y_true, y_pred, elapsed_time = None, labels = None):
    #
    if labels is None:
        labels = [x for x in numpy.unique(y_true)]
        print(labels)
    #
    os.makedirs(results_dir, exist_ok = True)
    f_results = open(f'{results_dir}/{filename_prefix}.txt', 'wt')
    try:
        _cm_ = confusion_matrix(y_true, y_pred, labels = labels)
    except:
        print(numpy.unique(y_true))
        print(numpy.unique(y_pred))
        f_results.write('Impossible to compute the results because all samples are in one class\n')
        f_results.close()
        return
        
    for i in range(_cm_.shape[0]):
        for j in range(_cm_.shape[1]):
            f_results.write(' %10d' % _cm_[i, j])
        f_results.write('\n')
    f_results.write('\n')
    print(classification_report(y_true, y_pred), file = f_results)
    if elapsed_time is not None:
        f_results.write('\n')
        f_results.write('\n')
        f_results.write('running time in seconds: %.3f\n' % elapsed_time)
        f_results.write('\n')
    f_results.close()
    #
    fig, axes = pyplot.subplots(nrows = 1, ncols = 2, figsize = (16, 7))
    #fig.suptitle(title)
    #
    ConfusionMatrixDisplay.from_estimator(estimator = MyArgmaxForPredictedLabels(classes_ = labels),
                          X = y_pred, y = y_true, # display_labels = labels,
                          normalize = 'true', ax = axes[0],
                          cmap = pyplot.cm.Blues) #, colorbar = False)
    #
    ConfusionMatrixDisplay.from_estimator(estimator = MyArgmaxForPredictedLabels(classes_ = labels),
                          X = y_pred, y = y_true, # display_labels = labels,
                          normalize = 'pred', ax = axes[1],
                          cmap = 'Oranges') #, colorbar = False)
    #
    pyplot.tight_layout()
    pyplot.savefig(f'{results_dir}/{filename_prefix}.svg', format = 'svg')
    pyplot.savefig(f'{results_dir}/{filename_prefix}.png', format = 'png')
    fig.clear()
    pyplot.close(fig)
    del fig
# ---------------------------------------------------------------------------------------------------
