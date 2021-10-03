import os
import sys
import numpy

from matplotlib import pyplot

from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix

# ---------------------------------------------------------------------------------------------------
class MyArgmaxForPredictedLabels(BaseEstimator):
    def __init__(self, threshold = 0.5):
        self.threshold = threshold
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
def save_results(results_dir, filename_prefix, y_true, y_pred):
    os.makedirs(results_dir, exist_ok = True)
    f_results = open(f'{results_dir}/{filename_prefix}.txt', 'wt')
    _cm_ = confusion_matrix(y_true, y_pred)
    for i in range(_cm_.shape[0]):
        for j in range(_cm_.shape[1]):
            f_results.write(' %10d' % _cm_[i, j])
        f_results.write('\n')
    f_results.write('\n')
    print(classification_report(y_true, y_pred), file = f_results)
    f_results.close()
    #
    fig, axes = pyplot.subplots(nrows = 1, ncols = 2, figsize = (16, 7))
    #fig.suptitle(title)
    #
    plot_confusion_matrix(estimator = MyArgmaxForPredictedLabels(),
                          X = y_pred, y_true = y_true,
                          normalize = 'true', ax = axes[0], cmap = 'Blues', colorbar = False)
    #
    plot_confusion_matrix(estimator = MyArgmaxForPredictedLabels(),
                          X = y_pred, y_true = y_true,
                          normalize = 'pred', ax = axes[1], cmap = 'Oranges', colorbar = False)
    #
    pyplot.tight_layout()
    pyplot.savefig(f'{results_dir}/{filename_prefix}.svg', format = 'svg')
    pyplot.savefig(f'{results_dir}/{filename_prefix}.png', format = 'png')
    fig.clear()
    pyplot.close(fig)
    del fig
# ---------------------------------------------------------------------------------------------------
