import numpy as np



def prox(v, beta, k):
    # argmin_v .5 ||x - v||_F^2 + 1 / (2 * beta) k_support_squared(v)
    d = len(v)
    z = [np.inf]
    z.extend(sorted(abs(v), reverse=True))
    z.append(-np.inf)
    sign_v = np.sign(v)
    argsort_v = np.argsort(-np.abs(v))
    z = np.array(z)
    condition = False
    l = k + 1
    T0 = np.cumsum(z[:(k + 1)][::-1])
    t1 = 0.0
    while not condition and (l <= d):
        r = 0
        t1 += z[l]
        while (r < k) and not condition:
            T = T0[r] + t1
            tmp = T / (l - k + (beta + 1) * r + beta + 1)
            cond_1 = z[k - r - 1] > (beta + 1) * tmp
            cond_2 = z[k - r] <= (beta + 1) * tmp
            cond_3 = z[l] > tmp
            cond_4 = z[l + 1] <= tmp
            condition = cond_1 and cond_2 and cond_3 and cond_4
            r += 1
        l += 1
    r -= 1
    l -= 1

    q = np.zeros(d)
    for i in range(k - r):
        q[i] = z[i + 1] * beta / (beta + 1)
    for i in range(k - r, l):
        q[i] = z[i + 1] - tmp
    q_re_sorted = np.zeros(d)
    for i in range(d):
        q_re_sorted[argsort_v[i]] = q[i]
    q_re_sorted = q_re_sorted * sign_v
    return q_re_sorted


def dual(v, k):
    return .5 * np.sum(sorted(v ** 2, reverse=True)[:k])


def value(v, k):

    d = len(v)
    if k >= d:
        return .5 * np.sum(v **2)
    elif k == 1:
        return .5 * np.sum(np.abs(v)) ** 2
    else:
        # nontrivial case
        z = [np.inf]
        z.extend(sorted(abs(v), reverse=True))
        z = np.array(z)
        condition = False
        r = 0
        partial_sum = np.sum(z[(k+1):])
        while not condition and (r < k):
            partial_sum += z[k-r]
            tmp = partial_sum / (r+1)
            cond_1 = z[k - r - 1] > tmp
            cond_2 = z[k - r] <= tmp
            condition = cond_1 and cond_2
            r += 1
        r -= 1
        squared_norm = 0.0
        for i in range(k - r):
            squared_norm += z[i + 1] ** 2
        l1_part = 0.0
        for i in range(k - r, d):
            l1_part += z[i + 1]
        squared_norm += l1_part ** 2 / (r+1)
        return .5 * squared_norm


def subgrad(v, k):

    argsort_v = np.argsort(-np.abs(v))
    d = len(v)
    if k >= d:
        return v
    elif k == 1:
        return np.sum(np.abs(v)) * np.sign(v)
    else:
        # nontrivial case
        z = [np.inf]
        z.extend(sorted(abs(v), reverse=True))
        z = np.array(z)
        condition = False
        r = 0
        partial_sum = np.sum(z[(k+1):])
        while not condition and (r < k):
            partial_sum += z[k-r]
            tmp = partial_sum / (r+1)
            cond_1 = z[k - r - 1] > tmp
            cond_2 = z[k - r] <= tmp
            condition = cond_1 and cond_2
            r += 1
        r -= 1

        q = np.zeros(d)
        for i in range(k - r):
            q[i] = z[i + 1]
        partial_l1 = np.sum(z[(k-r):])
        for i in range(k - r, d):
            if z[i + 1] > 0.:
                q[i] = partial_l1 / (r+1)
            else:
                break
        q_re_sorted = np.zeros(d)
        for i in range(d):
            q_re_sorted[argsort_v[i]] = q[i]
        q_re_sorted = q_re_sorted * np.sign(v)
        return q_re_sorted


def first_order_optimality(v, gradient, k, regularization_parameter):

    argsort_v = np.argsort(-np.abs(v))
    d = len(v)
    if k >= d:
        return np.linalg.norm(gradient + regularization_parameter * np.sqrt(np.linalg.norm(v)) * v)
    elif k == 1:
        return np.linalg.norm(gradient + regularization_parameter * np.sqrt(np.linalg.norm(v)) * np.sign(v))
    else:
        # nontrivial case
        z = [np.inf]
        z.extend(sorted(abs(v), reverse=True))
        z = np.array(z)
        condition = False
        r = 0
        partial_sum = np.sum(z[(k+1):])
        while not condition and (r < k):
            partial_sum += z[k-r]
            tmp = partial_sum / (r+1)
            cond_1 = z[k - r - 1] > tmp
            cond_2 = z[k - r] <= tmp
            condition = cond_1 and cond_2
            r += 1
        r -= 1
        index_type = np.zeros(len(v))
        squared_norm = 0.0
        for i in range(k - r):
            index_type[i] = 2
            squared_norm += z[i + 1] ** 2
        l1_part = 0.0
        for i in range(k - r, d):
            index_type[i] = 1
            l1_part += z[i + 1]

        index_type_re_sorted = np.zeros(len(v))
        for i in range(d):
            index_type_re_sorted[argsort_v[i]] = index_type[i]

        l2_component = gradient * (index_type_re_sorted == 2) + \
            regularization_parameter * np.sqrt(squared_norm) * v * (index_type_re_sorted == 2)
        l1_component = gradient * (index_type_re_sorted == 1) + \
            regularization_parameter * l1_part * np.sign(v) * (index_type_re_sorted == 1) / (r+1)
        zero_component = gradient * (index_type_re_sorted == 0) * (r+1) / (l1_part * regularization_parameter)


        #print 'l2 = %2.2e l1 = %2.2e 00 = %2.2e' %(np.linalg.norm(l2_component), np.linalg.norm(l1_component), np.max(np.abs(zero_component)))

        return np.sqrt(np.linalg.norm(l2_component) ** 2 + np.linalg.norm(l1_component) + np.max(np.abs(zero_component)))

