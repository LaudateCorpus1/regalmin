import numpy as np
import json
import matplotlib.pyplot as plt
import argparse

import fbpca
from sklearn import preprocessing

from regularized_pca import RegularizedLeastSquares
from deflated_power_iteration import DeflatedPCA


DEFAULT_PATH = '/Users/emilerichard/Documents/data/gene_expression/data_matrix.npy'


def evaluate(input_matrix, left_factor, right_factor):
    diff_norm = np.linalg.norm(input_matrix - left_factor.dot(right_factor.T), 'fro')\
                / np.linalg.norm(input_matrix, 'fro')
    for col in range(left_factor.shape[1]):
        left_factor[:, col] /= np.linalg.norm(left_factor[:, col])
        right_factor[:, col] /= np.linalg.norm(right_factor[:, col])

    explained_variance = np.sum(np.diag(left_factor.T.dot(input_matrix).dot(right_factor)))
    left_sparsity = np.sum(left_factor != 0.0) / np.float(np.prod(left_factor.shape))
    right_sparsity = np.sum(right_factor != 0.) / np.float(np.prod(right_factor.shape))
    print 'difference |X-UV^T|_F = {} explained variance Trace(U^T X V) = {}, nnz = {}'.\
        format(diff_norm, explained_variance, right_sparsity)
    result = {
        'distance': diff_norm,
        'variance': explained_variance,
        'left_sparsity': left_sparsity,
        'right_sparsity': right_sparsity
    }
    return result


if __name__ == "__main__":

    NB_VARIABLES_TO_USE = 2000
    q = 50  # 200
    r = 5
    max_iter_full_factor = 4000
    max_iter_power_method = 50
    default_lambda = .5
    beta = 8.0

    parser = argparse.ArgumentParser()
    parser.add_argument('--path',\
        default=DEFAULT_PATH,\
        help ='run load_gene_expression_data.py first and give the output file path as argument here')
    args = parser.parse_args()
    path = args.path
    mat = np.load(path)

    mat_scaled = preprocessing.scale(mat, with_std=False)
    mat = mat_scaled[:, np.argsort(-np.sum(mat**2, axis=0))[:NB_VARIABLES_TO_USE]]

    kq_parameters = {
        'r': r,
        'left_regularizer': 'l2',
        'right_regularizer': 'k-support',
        'regularization': default_lambda,
        #'display_objective': True,
        'q': q,
        'max_iter': max_iter_full_factor,
        'tolerance': .01
        }
    deflation_l1_parameters = {
        'r': r,
        'q': .0,
        'right_regularizer': 'l1',
        'regularization': default_lambda,
        'max_iter': max_iter_power_method
        }

    range_lambda = [2. ** x for x in range(-10, 4)]
    range_lambda = [2. ** x for x in range(4, 6)]
    kq_res = {}
    def_l1_res = {}
    full_res = {}

    u, s, v = fbpca.pca(mat, r)
    u = u.dot(np.diag(np.sqrt(s)))
    v = v.T.dot(np.diag(np.sqrt(s)))
    full_res = evaluate(mat, u, v)

    for regularization_parameter in range_lambda:
        print 'lambda = %2.2e' % regularization_parameter
        kq_parameters['regularization'] = regularization_parameter
        deflation_l1_parameters['regularization'] = regularization_parameter

        kq_spca = RegularizedLeastSquares(mat, kq_parameters)
        kq_spca.optimize()

        deflated_l1_pca = DeflatedPCA(mat, deflation_l1_parameters)
        deflated_l1_pca.compute()

        kq_eval = evaluate(mat, kq_spca.regularized_left.matrix, kq_spca.regularized_right.matrix)
        kq_res[regularization_parameter] = (kq_eval['right_sparsity'], kq_eval['distance'], kq_eval['variance'])

        l1_eval = evaluate(mat, deflated_l1_pca.left_factor, deflated_l1_pca.right_factor)
        def_l1_res[regularization_parameter] = (l1_eval['right_sparsity'], l1_eval['distance'], l1_eval['variance'])

    local_file_name = 'kq_local_46.json'
    with open(local_file_name, 'w') as local_file:
        local_file.write(json.dumps(kq_res))
    local_file_name = 'l1_local_46.json'
    with open(local_file_name, 'w') as local_file:
        local_file.write(json.dumps(def_l1_res))
    local_file_name = 'full_local_46.json'
    with open(local_file_name, 'w') as local_file:
        local_file.write(json.dumps(full_res))
    """
    s_def_l1 = [x[0] for x in def_l1_res.values() if not np.isnan(x[1])]
    s_kq = [x[0] for x in kq_res.values()]
    dis_def_l1 = [x[1] for x in def_l1_res.values() if x[1] is not np.isnan(x[1])]
    dis_kq = [x[1] for x in kq_res.values()]

    plt.plot(np.sort(s_kq), [dis_kq[i] for i in np.argsort(s_kq)], 'ro-', label='Proposed')
    plt.plot(np.sort(s_def_l1),  [dis_def_l1[i] for i in np.argsort(s_def_l1)], 'bx-', label='Witten et al.')
    plt.xlabel('ratio of non-zeros in the right factor')
    plt.ylabel('normalized distance')
    plt.legend()
    plt.show()
    """

