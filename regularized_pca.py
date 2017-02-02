import numpy as np
import fbpca
import vector_squared_k_support
import l1


class PsdLeastSquares:

    def __init__(self, input_matrix, factor):
        self.input_matrix = input_matrix
        self.factor = factor

    def residual(self):
        return self.factor.dot(self.factor.T) - self.input_matrix

    def value(self):
        return .25 * np.sum(self.residual() ** 2)

    def gradient(self):
        return self.residual().dot(self.factor)

    def lipschitz(self):
        return np.linalg.norm(self.factor.T.dot(self.factor))


class LeastSquares:

    def __init__(self, input_matrix, left_factor, right_factor):
        self.input_matrix = input_matrix
        self.left_factor = left_factor
        self.right_factor = right_factor

    def residual(self):
        return self.left_factor.dot(self.right_factor.T) - self.input_matrix

    def value(self):
        return .5 * np.sum(self.residual() ** 2)

    def gradient_left(self):
        return self.residual().dot(self.right_factor)

    def gradient_right(self):
        return self.residual().T.dot(self.left_factor)

    def lipschitz_left(self):
        return np.linalg.norm(self.right_factor.T.dot(self.right_factor))

    def lipschitz_right(self):
        return np.linalg.norm(self.left_factor.T.dot(self.left_factor))


class SquarredL2:

    def __init__(self, matrix):
        self.matrix = matrix

    def value(self):
        return .5 * np.sum(self.matrix ** 2)

    def dual(self):
        return .5 * np.sum(self.matrix ** 2)

    def prox(self, regularizer):
        return self.matrix / (1 + regularizer)

    def first_order_optimality(self, loss_gradient, regularization_parameter):
        regularized_gradient = loss_gradient + regularization_parameter * self.matrix
        return np.linalg.norm(regularized_gradient) / np.sqrt(np.prod(loss_gradient.shape))


class L1:

    def __init__(self, matrix):
        self.matrix = matrix

    def value(self):
        return l1.value(self.matrix)

    def dual(self):
        return l1.dual(self.matrix)

    def prox(self, regularization_parameter):
        abs_part = np.abs(self.matrix) - regularization_parameter
        return np.sign(self.matrix) * (abs_part * (abs_part > 0.))

    def first_order_optimality(self, loss_gradient, regularization_parameter):
        regularized_gradient = loss_gradient + regularization_parameter * np.sign(self.matrix)
        orthogonal_component = loss_gradient * (self.matrix == 0)
        return max(np.linalg.norm(regularized_gradient), np.max(np.abs(orthogonal_component)) /
                   regularization_parameter) / np.sqrt(np.prod(loss_gradient.shape))


class NonNegativeL1:

    def __init__(self, matrix):
        self.matrix = matrix

    def value(self):
        return np.sum(self.matrix * (self.matrix > 0.0))

    def dual(self):
        return self.matrix * (self.matrix > 0.0)

    def prox(self, regularization_parameter):
        pos_part = self.matrix - regularization_parameter
        return pos_part * (pos_part > 0.)

    def first_order_optimality(self, loss_gradient, regularization_parameter):
        regularized_gradient = loss_gradient + regularization_parameter * (self.matrix) * (self.matrix > 0.0)
        orthogonal_component = loss_gradient * (self.matrix <= 0)
        return max(np.linalg.norm(regularized_gradient), np.max(np.abs(orthogonal_component)) /
                   regularization_parameter) / np.sqrt(np.prod(loss_gradient.shape))


class SquaredKSupport:

    def __init__(self, matrix, k):
        self.matrix = matrix
        self.k = k

    def value(self):
        # half the sum of squared k-support norms of columns
        squared_norm = 0.0
        for col in xrange(self.matrix.shape[1]):
            squared_norm += vector_squared_k_support.value(self.matrix[:, col], self.k)
        return .5 * squared_norm

    def dual(self):
        dual_norm_squared = 0.0
        for col_index in xrange(self.matrix.shape[1]):
            curr_dual = vector_squared_k_support.dual(self.matrix[:, col_index], self.k)
            if curr_dual > dual_norm_squared:
                dual_norm_squared = curr_dual
        return .5 * dual_norm_squared

    def prox(self, regularization_over_lipschitz):
        beta = .5 / regularization_over_lipschitz
        proximal = np.zeros(self.matrix.shape)
        for col_index in xrange(self.matrix.shape[1]):
            proximal[:, col_index] = vector_squared_k_support.prox(self.matrix[:, col_index], beta, self.k)
        return proximal

    def subgradient(self):
        proximal = np.zeros(self.matrix.shape)
        for col_index in xrange(self.matrix.shape[1]):
            proximal[:, col_index] = vector_squared_k_support.subgradient(self.matrix[:, col_index], self.k)
        return proximal

    def first_order_optimality(self, loss_gradient, regularization_parameter):
        first_order_condition = 0.0
        for col in xrange(self.matrix.shape[1]):
            val = vector_squared_k_support.first_order_optimality\
                (self.matrix[:, col], loss_gradient[:, col], self.k, regularization_parameter)
            first_order_condition += val
        return first_order_condition * .5 / np.sqrt(np.prod(loss_gradient.shape))


class RegularizedLeastSquares:

    def __init__(self, input_matrix, parameter):
        if 'psd' in parameter:
            d_init, v_init = fbpca.eigenn(input_matrix, parameter['r'])
            factor = v_init.dot(np.diag(np.sqrt(d_init)))
            self.loss = PsdLeastSquares(input_matrix, factor)
            if parameter['regularizer'] == 'k-support':
                self.regularized_factor = SquaredKSupport(factor, parameter['k'])
            elif parameter['regularizer'] == 'l1':
                self.regularized_factor = L1(factor)
            elif parameter['regularizer'] == 'pos':
                self.regularized_factor = NonNegativeL1(factor)
            else:
                self.regularized_factor = SquarredL2(factor)

        else:
            u_init, sigma_init, v_init = fbpca.pca(input_matrix, parameter['r'])
            left_factor = u_init.dot(np.diag(np.sqrt(sigma_init)))
            right_factor = v_init.T.dot(np.diag(np.sqrt(sigma_init)))
            self.loss = LeastSquares(input_matrix, left_factor, right_factor)

            if parameter['left_regularizer'] == 'k-support':
                self.regularized_left = SquaredKSupport(left_factor, parameter['k'])
            elif parameter['left_regularizer'] == 'l1':
                self.regularized_left = L1(left_factor)
            elif parameter['left_regularizer'] == 'pos':
                self.regularized_left = NonNegativeL1(left_factor)
            else:
                self.regularized_left = SquarredL2(left_factor)

            if parameter['right_regularizer'] == 'k-support':
                self.regularized_right = SquaredKSupport(right_factor, parameter['q'])
            elif parameter['right_regularizer'] == 'l1':
                self.regularized_right = L1(right_factor)
            elif self.parameter['right_regularizer'] == 'pos':
                self.right_factor = NonNegativeL1(right_factor)
            else:
                self.regularized_right = SquarredL2(right_factor)

        self.parameter = parameter
        self.objective = [np.inf]
        self.first_order_condition = [np.inf]

    def update_objective(self, foc):
        loss_value = self.loss.value()
        reg_param = self.parameter['regularization']
        if 'psd' in self.parameter:
            penalty_value = reg_param * (self.regularized_factor.value())
        else:
            penalty_value = reg_param * (self.regularized_left.value() + self.regularized_right.value())
        objective_value = loss_value + penalty_value
        if (len(self.objective) > 0) and (objective_value > self.objective[-1]):
                direction = '^'
        else:
                direction = 'v'

        if 'psd' in self.parameter:
            print 'obj %2.2e %s |g| %2.2e %s' % \
                  (objective_value, direction,\
                   foc['value'], foc['direction'])
        else:
            print 'obj %2.2e %s |g<| %2.2e %s |g>| %2.2e %s' % \
                  (objective_value, direction,\
                   foc['left'], foc['left_direction'],\
                   foc['right'], foc['right_direction'])
        self.objective.append(objective_value)

    def update_factor(self):
        grad = self.loss.gradient()
        lipschitz = self.loss.lipschitz()

        self.regularized_factor.matrix -= grad / lipschitz
        self.regularized_factor.matrix = self.regularized_factor.prox(self.parameter['regularization'] / lipschitz)
        self.loss.factor = self.regularized_factor.matrix

    def update_left(self):
        grad = self.loss.gradient_left()
        lipschitz = self.loss.lipschitz_left()

        self.regularized_left.matrix -= grad / lipschitz
        self.regularized_left.matrix = self.regularized_left.prox(self.parameter['regularization'] / lipschitz)
        self.loss.left_factor = self.regularized_left.matrix

    def update_right(self):
        grad = self.loss.gradient_right()
        lipschitz = self.loss.lipschitz_right()
        self.regularized_right.matrix -= grad / lipschitz
        self.regularized_right.matrix = self.regularized_right.prox(self.parameter['regularization'] / lipschitz)
        self.loss.right_factor = self.regularized_right.matrix

    def optimize(self):
        if 'psd' in self.parameter:
            foc_list = [np.inf]
        else:
            foc_right = [np.inf]
            foc_left = [np.inf]
        for _ in xrange(self.parameter['max_iter']):
            if 'psd' in self.parameter:
                self.update_factor()
                foc_value = self.regularized_factor.first_order_optimality \
                    (self.loss.gradient(), self.parameter['regularization'])

                foc_list.append(foc_value)

                if foc_value < self.parameter['tolerance']:
                    print 'First order optimality -approximately- met:'
                    print '1st order condition %2.6f' % foc_value
                    break

                elif 'display_objective' in self.parameter:
                    foc_dictionary = {'value': foc_value}
                    if foc_value < foc_list[-2]:
                        foc_dictionary['direction'] = 'v'
                    else:
                        foc_dictionary['direction'] = '^'

                    self.update_objective(foc_dictionary)
            else:
                self.update_left()
                left_foc = self.regularized_left.first_order_optimality\
                    (self.loss.gradient_left(), self.parameter['regularization'])

                foc_left.append(left_foc)
                self.update_right()
                right_foc = self.regularized_right.first_order_optimality\
                    (self.loss.gradient_right(), self.parameter['regularization'])

                foc_right.append(right_foc)
                if (left_foc < self.parameter['tolerance']) and (right_foc < self.parameter['tolerance']):
                    print 'First order optimality -approximately- met on both left and right factors:'
                    print 'left 1st order condition %2.6f' % left_foc
                    print 'right 1st order condition %2.6f' % right_foc
                    break

                elif 'display_objective' in self.parameter:
                    foc = {'left': left_foc, 'right': right_foc}
                    if left_foc < foc_left[-2]:
                        foc['left_direction'] = 'v'
                    else:
                        foc['left_direction'] = '^'
                    if right_foc < foc_right[-2]:
                        foc['right_direction'] = 'v'
                    else:
                        foc['right_direction'] = '^'
                    self.update_objective(foc)
        if 'display_objective' in self.parameter:
            if 'psd' in self.parameter:
                import matplotlib.pyplot as plt
                plt.subplot(121)
                plt.loglog(self.objective - np.min(self.objective) + 1e-6)
                plt.xlabel('steps')
                plt.ylabel('objective - min(objective)')
                plt.grid()

                plt.subplot(122)
                plt.loglog(foc_list - np.min(foc_list) + 1e-6)
                plt.xlabel('steps')
                plt.ylabel('|gradient| - min(|gradient|)')
                plt.grid()
                plt.show()

            else:
                import matplotlib.pyplot as plt
                plt.subplot(131)
                plt.loglog(self.objective - np.min(self.objective) + 1e-6)
                plt.xlabel('steps')
                plt.ylabel('objective - min(objective)')
                plt.grid()

                plt.subplot(132)
                plt.loglog(foc_left - np.min(foc_left) + 1e-6)
                plt.xlabel('steps')
                plt.ylabel('|left gradient| - min(|left gradient|)')
                plt.grid()

                plt.subplot(133)
                plt.loglog(foc_right - np.min(foc_right) + 1e-6)
                plt.xlabel('steps')
                plt.ylabel('|right gradient| - min(|right gradient|)')
                plt.grid()
                plt.show()
