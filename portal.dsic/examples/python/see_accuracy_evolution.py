import os
import sys
import numpy
from matplotlib import pyplot


if __name__ == '__main__':
    task = 'classification' # 'prediction'
    clustering = 'kmeans' # 'gmm'
    results_dir = 'results'


    for i in range(len(sys.argv)):
        if sys.argv[i] == '--classification':
            task = 'classification'
        elif sys.argv[i] == '--prediction':
            task = 'prediction'
        elif sys.argv[i] == '--kmeans':
            clustering = 'kmeans'
        elif sys.argv[i] == '--gmm':
            clustering = 'gmm'
        elif sys.argv[i] == '--results-dir':
            results_dir = sys.argv[i + 1]

    if clustering == 'kmeans':
        prefix = f'{task}-results'
    elif clustering == 'gmm':
        prefix = f'{clustering}-{task}-results'
    else:
        raise Exception(f'Invalid clustering type: {clustering}')
    suffix = '.txt'

    def extract_num_clusters(filename):
        parts = filename.split(sep = '-')
        str_n = parts[-1].split(sep = '.')[0]
        return int(str_n)

    def get_accuracy(filename):
        accuracy = None
        f = open(filename, 'rt')
        for line in f:
            parts = line.split()
            if len(parts) == 3 and parts[0] == 'accuracy':
                accuracy = float(parts[1])
        f.close()
        if accuracy is None:
            raise Exception(f'accuracy not found in {filename}')
        return accuracy
       
    data = list()
    for root, dirs, files in os.walk(results_dir):
        #print(root, len(dirs), len(files))
        filenames = [f for f in filter(lambda fname: fname.startswith(prefix) and fname.endswith(suffix), files)]
        for filename in filenames:
            data.append((extract_num_clusters(filename), get_accuracy(f'{root}/{filename}')))

    data.sort(key = lambda x: x[0])

    data = numpy.array(data)

    filename = f'{root}/accuracy-{clustering}-{task}'

    fig, axes = pyplot.subplots(nrows = 1, ncols = 1, figsize = (9, 6))
    #
    axis = axes
    axis.set_facecolor('#eefffc')
    axis.plot(data[:,0], data[:, 1], 'ro-', alpha = 1.0)
    axis.grid()
    axis.set_xlabel('Number of clusters')
    axis.set_ylabel('Accuracy')
    axis.set_title('Accuracy evolution according to the number of clusters')
    pyplot.tight_layout()
    for file_format in ['png', 'svg']:
        pyplot.savefig(f'{filename}.{file_format}', format = file_format)
        print(f'{filename}.{file_format} created')
    del fig
