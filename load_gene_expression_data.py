import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',\
        help = 'download data from http://ccb.nki.nl/data/ZipFiles295Samples.zip \n '\
               'give the unzipped folder s path as argument here')
    args = parser.parse_args()
    path = args.path

    fi = {}
    all_data = []
    raw_matrix = []
    for i in range(1, 7):
        filename = '{}Table_NKI_295_{}.txt'.format(path, i)
        fi[i] = [x.split('\t')[:-1] for x in open(filename, 'r').read().split('\n')]
        all_data.extend([x for x in fi[i]])
        if i == 6:  # for some reason the 6th file is configured differently
            fi[i] = fi[i][:-1]
        raw_matrix.append(np.array([[float(y) for y in x[2:252:5]] for x in fi[i][2:]]))

    mat = np.concatenate(raw_matrix, axis =1).T
    mat[np.isnan(mat)] = 0

    data_matrix_file = '{}data_matrix.npy'.format(path)
    print 'output will be written at %s' %data_matrix_file
    np.save(open(data_matrix_file, 'w'), mat)
