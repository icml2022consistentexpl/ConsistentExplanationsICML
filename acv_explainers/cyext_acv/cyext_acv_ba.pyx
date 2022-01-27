# distutils: language = c++

from libcpp.vector cimport vector
import numpy as np
cimport numpy as np
ctypedef np.float64_t double
cimport cython
from scipy.special import comb
import itertools
from tqdm import tqdm
from cython.parallel cimport prange, parallel, threadid
cimport openmp

cdef extern from "<algorithm>" namespace "std" nogil:
     iter std_remove "std::remove" [iter, T](iter first, iter last, const T& val)
     iter std_find "std::find" [iter, T](iter first, iter last, const T& val)

cdef extern from "limits.h":
    unsigned long ULONG_MAX



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
# @cython.cdivision(True)
cpdef shap_values_leaves_pa(const double[:, :] X,
    const long[:, :, :, :] data,
    const double[:, :, :] values,
    const double[:, :, :, :] partition_leaves_trees,
    const long[:, :] leaf_idx_trees,
    const long[::1] leaves_nb,
    const long scaling,
    const vector[vector[vector[long]]] node_idx_trees,
    const vector[vector[long]] C, int num_threads):


    cdef unsigned int d = values.shape[2]
    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]
    cdef unsigned int n_trees = values.shape[0]
    cdef unsigned int max_leaves = partition_leaves_trees.shape[1]

    cdef double[:, :, :] phi
    phi = np.zeros((N, m, d))

    cdef double[ :, :, :,  :, :] phi_b
    phi_b = np.zeros((max_leaves, 2**scaling, N, m, d))

    cdef long[:, :, :, ::1] S
    S = np.zeros((m, 2**scaling, max_leaves, m), dtype=np.int)

    cdef unsigned int a_it, nb_leaf, va,  counter, ci, cj, ite, pow_set_size, nv, ns, add
    cdef unsigned int b, leaf_numb, i, s, j, i1, i2, l, na_bool, o_all, csi, cs
    cdef double lm

    cdef vector[vector[long]] C_buff, va_id, buff
    cdef vector[long] coal_va, remove_va, node_id_b, buff_l
    buff.resize(max_leaves)
    for i in range(max_leaves):
        buff[i].resize(1)

    cdef vector[vector[vector[long]]] node_id_v2, C_b, Sm, node_id
    node_id_v2.resize(max_leaves)
    C_b.resize(max_leaves)
    Sm.resize(max_leaves)
    node_id.resize(max_leaves)

    cdef vector[vector[vector[vector[long]]]] node_id_va
    node_id_va.resize(m)
    for i in range(m):
        node_id_va[i].resize(max_leaves)

    cdef int S_size, va_size,  b_it, nv_bool

    cdef long set_size

    if C[0].size() != 0:
        C_buff = C
        for ci in range(C.size()):
            for cj in range(C[ci].size()):
                coal_va.push_back(C[ci][cj])

        for i in range(m):
            if not std_find[vector[long].iterator, long](coal_va.begin(), coal_va.end(), i) != coal_va.end():
                remove_va.push_back(i)
                buff[0][0] = i
                va_id.push_back(buff[0])
            else:
                for ci in range(C_buff.size()):
                    if (std_find[vector[long].iterator, long](C_buff[ci].begin(), C_buff[ci].end(), i) != C_buff[ci].end()):
                        va_id.push_back(C_buff[ci])
                        std_remove[vector[vector[long]].iterator, vector[long]](C_buff.begin(), C_buff.end(), C_buff[ci])
                        C_buff.pop_back()
                        break

    else:
        for i in range(m):
            buff[0][0] = i
            va_id.push_back(buff[0])

    cdef double p_s, p_si, coef, coef_0
    cdef double[:, :, :] lm_s, lm_si
    lm_s = np.zeros((va_id.size(), max_leaves, 2**scaling))
    lm_si = np.zeros((va_id.size(), max_leaves, 2**scaling))

    for b in range(n_trees):
        for leaf_numb in range(phi_b.shape[0]):
            for counter in range(phi_b.shape[1]):
                for i in range(N):
                    for j in range(m):
                        for i2 in range(d):
                            phi_b[leaf_numb, counter, i, j, i2] = 0
        nb_leaf = leaves_nb[b]
        for leaf_numb in range(nb_leaf):
            node_id_v2[leaf_numb].clear()
            lm = 0
            for i in range(data.shape[2]):
                a_it = 1
                for s in range(data.shape[3]):
                    a_it = a_it * data[b, leaf_numb, i, s]

                lm = lm + a_it

            if C[0].size() != 0:
                C_b[leaf_numb] = C
                for nv in range(node_idx_trees[b][leaf_numb].size()):
                    add = 0
                    for ns in range(remove_va.size()):
                        if node_idx_trees[b][leaf_numb][nv] == remove_va[ns]:
                            add = 1
                            buff[leaf_numb][0] = node_idx_trees[b][leaf_numb][nv]
                            node_id_v2[leaf_numb].push_back(buff[leaf_numb])
                            break
                    if add == 0:
                        for ci in range(C_b[leaf_numb].size()):
                            for cj in range(C_b[leaf_numb][ci].size()):
                                if C_b[leaf_numb][ci][cj] == node_idx_trees[b][leaf_numb][nv]:
                                    add = 1
                                    node_id_v2[leaf_numb].push_back(C_b[leaf_numb][ci])
                                    break
                            if add == 1:
                                std_remove[vector[vector[long]].iterator, vector[long]](C_b[leaf_numb].begin(), C_b[leaf_numb].end(), C_b[leaf_numb][ci])
                                C_b[leaf_numb].pop_back()
                                break
                node_id[leaf_numb] = node_id_v2[leaf_numb]
            else:
                node_id[leaf_numb].clear()
                for i in range(node_idx_trees[b][leaf_numb].size()):
                    buff[leaf_numb][0] = node_idx_trees[b][leaf_numb][i]
                    node_id[leaf_numb].push_back(buff[leaf_numb])


            for va in range(va_id.size()):
                node_id_va[va][leaf_numb] = node_id[leaf_numb]
                if not std_find[vector[vector[long]].iterator, vector[long]](node_id_va[va][leaf_numb].begin(), node_id_va[va][leaf_numb].end(), va_id[va]) != node_id_va[va][leaf_numb].end():
                    continue

                std_remove[vector[vector[long]].iterator, vector[long]](node_id_va[va][leaf_numb].begin(), node_id_va[va][leaf_numb].end(), va_id[va])
                node_id_va[va][leaf_numb].pop_back()

                set_size = node_id_va[va][leaf_numb].size()
                pow_set_size = 2**set_size

                for counter in range(0, pow_set_size):
                    va_size = 0
                    S_size = 0
                    for ci in range(set_size):
                        if((counter & (1 << ci)) > 0):
                            for cj in range(node_id_va[va][leaf_numb][ci].size()):
                                S[va, counter, leaf_numb, S_size] = node_id_va[va][leaf_numb][ci][cj]
                                S_size = S_size + 1
                            va_size = va_size + 1

                    lm_s[va, leaf_numb, counter] = 0
                    lm_si[va, leaf_numb, counter] = 0
                    for i in range(data.shape[2]):
                        b_it = 1
                        for s in range(S_size):
                            b_it = b_it * data[b, leaf_numb, i, S[va, counter, leaf_numb, s]]

                        lm_s[va, leaf_numb, counter] = lm_s[va, leaf_numb, counter] + b_it

                        for nv in range(va_id[va].size()):
                            b_it = b_it * data[b, leaf_numb, i, va_id[va][nv]]

                        lm_si[va, leaf_numb, counter] = lm_si[va, leaf_numb, counter] + b_it

                    for i in range(N):

                        csi = 0
                        cs = 0

                        o_all = 0
                        for s in range(S_size):
                            o_all = o_all + 1
                        nv_bool = 0
                        for nv in range(va_id[va].size()):
                            nv_bool = nv_bool + 1

                        coef = 0
                        for l in range(1, m - set_size):
                            coef = coef + (1.*binomialC(m - set_size - 1, l))/binomialC(m - 1, l + va_size) if binomialC(m - 1, l + va_size) !=0 else 0

                        coef_0 = 1./binomialC(m-1, va_size) if binomialC(m-1, va_size) !=0 else 0

                        if S_size == 0:
                            p_s = lm/data.shape[2]
                        else:
                            p_s = (cs * lm)/lm_s[va, leaf_numb, counter] if lm_s[va, leaf_numb, counter] !=0 else 0
                        p_si = (csi * lm)/lm_si[va, leaf_numb, counter] if lm_si[va, leaf_numb, counter] !=0 else 0

                        for nv in range(va_id[va].size()):
                            for i2 in range(d):
                                phi_b[leaf_numb, counter, i, va_id[va][nv], i2] += (coef_0 + coef) * (p_si - p_s) * values[b, leaf_idx_trees[b, leaf_numb], i2]

        for i in range(N):
            for j in range(m):
                for i2 in range(d):
                    for leaf_numb in range(phi_b.shape[0]):
                        for counter in range(phi_b.shape[1]):
                            phi[i, j, i2] += phi_b[leaf_numb, counter, i, j, i2]

    return np.asarray(phi)/m


cdef unsigned long binomialC(unsigned long N, unsigned long k) nogil:
    cdef unsigned long r
    r = _comb_int_long(N, k)
    if r != 0:
        return r

cdef unsigned long _comb_int_long(unsigned long N, unsigned long k) nogil:
    """
    Compute binom(N, k) for integers.
    Returns 0 if error/overflow encountered.
    """
    cdef unsigned long val, j, M, nterms

    if k > N or N == ULONG_MAX:
        return 0

    M = N + 1
    nterms = min(k, N - k)

    val = 1

    for j in range(1, nterms + 1):
        # Overflow check
        if val > ULONG_MAX // (M - j):
            return 0

        val *= M - j
        val //= j

    return val



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef leaves_cache(
    const double[:, :] data,
    const double[:, :, :] values,
    const double[:, :, :, :] partition_leaves_trees,
    const long[:, :] leaf_idx_trees,
    const long[::1] leaves_nb,
    const long scaling,
    const vector[vector[vector[long]]] node_idx_trees,
    const vector[vector[long]] C, int num_threads):


    cdef unsigned int d = values.shape[2]
    cdef unsigned int m = data.shape[1]
    cdef unsigned int n_trees = values.shape[0]
    cdef unsigned int max_leaves = partition_leaves_trees.shape[1]

    cdef long[:, :, :, ::1] S
    S = np.zeros((m, 2**scaling, max_leaves, m), dtype=np.int)

    cdef unsigned int a_it, nb_leaf, va,  counter, ci, cj, ite, pow_set_size, nv, ns, add
    cdef unsigned int b, leaf_numb, i, s, j, i1, i2, l, na_bool, o_all, csi, cs
    cdef double[:, :] lm
    lm = np.zeros((n_trees, max_leaves))

    cdef vector[vector[long]] C_buff, va_id, buff
    cdef vector[long] coal_va, remove_va, node_id_b, buff_l
    buff.resize(max_leaves)
    for i in range(max_leaves):
        buff[i].resize(1)

    cdef vector[vector[vector[long]]] node_id_v2, C_b, Sm, node_id
    node_id_v2.resize(max_leaves)
    C_b.resize(max_leaves)
    Sm.resize(max_leaves)
    node_id.resize(max_leaves)

    cdef vector[vector[vector[vector[long]]]] node_id_va
    node_id_va.resize(m)
    for i in range(m):
        node_id_va[i].resize(max_leaves)

    cdef int S_size, va_size,  b_it, nv_bool

    cdef long set_size

    if C[0].size() != 0:
        C_buff = C
        for ci in range(C.size()):
            for cj in range(C[ci].size()):
                coal_va.push_back(C[ci][cj])

        for i in range(m):
            if not std_find[vector[long].iterator, long](coal_va.begin(), coal_va.end(), i) != coal_va.end():
                remove_va.push_back(i)
                buff[0][0] = i
                va_id.push_back(buff[0])
            else:
                for ci in range(C_buff.size()):
                    if (std_find[vector[long].iterator, long](C_buff[ci].begin(), C_buff[ci].end(), i) != C_buff[ci].end()):
                        va_id.push_back(C_buff[ci])
                        std_remove[vector[vector[long]].iterator, vector[long]](C_buff.begin(), C_buff.end(), C_buff[ci])
                        C_buff.pop_back()
                        break

    else:
        for i in range(m):
            buff[0][0] = i
            va_id.push_back(buff[0])

    cdef double p_s, p_si, coef, coef_0
    cdef double[:, :, :, :] lm_s, lm_si
    lm_s = np.zeros((n_trees, va_id.size(), max_leaves, 2**scaling))
    lm_si = np.zeros((n_trees, va_id.size(), max_leaves, 2**scaling))

    for b in range(n_trees):
        nb_leaf = leaves_nb[b]
        for leaf_numb in prange(nb_leaf, nogil=True, schedule='dynamic'):
            node_id_v2[leaf_numb].clear()
            lm[leaf_numb] = 0
            for i in range(data.shape[0]):
                a_it = 0
                for s in range(data.shape[1]):
                    if (data[i, s] <= partition_leaves_trees[b, leaf_numb, s, 1]) and (data[i, s] > partition_leaves_trees[b, leaf_numb, s, 0]):
                        a_it = a_it + 1
                if a_it == data.shape[1]:
                    lm[b, leaf_numb] = lm[b, leaf_numb] + 1

            if C[0].size() != 0:
                C_b[leaf_numb] = C
                for nv in range(node_idx_trees[b][leaf_numb].size()):
                    add = 0
                    for ns in range(remove_va.size()):
                        if node_idx_trees[b][leaf_numb][nv] == remove_va[ns]:
                            add = 1
                            buff[leaf_numb][0] = node_idx_trees[b][leaf_numb][nv]
                            node_id_v2[leaf_numb].push_back(buff[leaf_numb])
                            break
                    if add == 0:
                        for ci in range(C_b[leaf_numb].size()):
                            for cj in range(C_b[leaf_numb][ci].size()):
                                if C_b[leaf_numb][ci][cj] == node_idx_trees[b][leaf_numb][nv]:
                                    add = 1
                                    node_id_v2[leaf_numb].push_back(C_b[leaf_numb][ci])
                                    break
                            if add == 1:
                                std_remove[vector[vector[long]].iterator, vector[long]](C_b[leaf_numb].begin(), C_b[leaf_numb].end(), C_b[leaf_numb][ci])
                                C_b[leaf_numb].pop_back()
                                break
                node_id[leaf_numb] = node_id_v2[leaf_numb]
            else:
                node_id[leaf_numb].clear()
                for i in range(node_idx_trees[b][leaf_numb].size()):
                    buff[leaf_numb][0] = node_idx_trees[b][leaf_numb][i]
                    node_id[leaf_numb].push_back(buff[leaf_numb])


            for va in range(va_id.size()):
                node_id_va[va][leaf_numb] = node_id[leaf_numb]
                if not std_find[vector[vector[long]].iterator, vector[long]](node_id_va[va][leaf_numb].begin(), node_id_va[va][leaf_numb].end(), va_id[va]) != node_id_va[va][leaf_numb].end():
                    continue

                std_remove[vector[vector[long]].iterator, vector[long]](node_id_va[va][leaf_numb].begin(), node_id_va[va][leaf_numb].end(), va_id[va])
                node_id_va[va][leaf_numb].pop_back()

                set_size = node_id_va[va][leaf_numb].size()
                pow_set_size = 2**set_size

                for counter in range(0, pow_set_size):
                    va_size = 0
                    S_size = 0
                    for ci in range(set_size):
                        if((counter & (1 << ci)) > 0):
                            for cj in range(node_id_va[va][leaf_numb][ci].size()):
                                S[va, counter, leaf_numb, S_size] = node_id_va[va][leaf_numb][ci][cj]
                                S_size = S_size + 1
                            va_size = va_size + 1

                    for i in range(data.shape[0]):
                        b_it = 0
                        for s in range(S_size):
                            if ((data[i, S[va, counter, leaf_numb, s]] <= partition_leaves_trees[b, leaf_numb, S[va, counter, leaf_numb, s], 1]) * (data[i, S[va, counter, leaf_numb, s]] > partition_leaves_trees[b, leaf_numb, S[va, counter, leaf_numb, s], 0])):
                                b_it = b_it + 1

                        if b_it == S_size:
                            lm_s[b, va, leaf_numb, counter] = lm_s[b, va, leaf_numb, counter] + 1

                            nv_bool = 0
                            for nv in range(va_id[va].size()):
                                if ((data[i, va_id[va][nv]] > partition_leaves_trees[b, leaf_numb, va_id[va][nv], 1]) or (data[i, va_id[va][nv]] <= partition_leaves_trees[b, leaf_numb, va_id[va][nv], 0])):
                                    nv_bool = nv_bool + 1
                                    continue

                            if nv_bool == 0:
                                lm_si[b, va, leaf_numb, counter] = lm_si[b, va, leaf_numb, counter] + 1

    return np.asarray(lm), np.asarray(lm_s), np.asarray(lm_si)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
# @cython.cdivision(True)
cpdef shap_values_leaves_cache(const double[:, :] X,
    const double[:, :] data,
    const double[:, :, :] values,
    const double[:, :, :, :] partition_leaves_trees,
    const long[:, :] leaf_idx_trees,
    const long[::1] leaves_nb,
    const double [:] lm,
    const double[:, :, :] lm_s,
    const double[:, :, :] lm_si,
    const long scaling,
    const vector[vector[vector[long]]] node_idx_trees,
    const vector[vector[long]] C, int num_threads):


    cdef unsigned int d = values.shape[2]
    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]
    cdef unsigned int n_trees = values.shape[0]
    cdef unsigned int max_leaves = partition_leaves_trees.shape[1]

    cdef double[:, :, :] phi
    phi = np.zeros((N, m, d))

    cdef double[ :, :, :,  :, :] phi_b
    phi_b = np.zeros((max_leaves, 2**scaling, N, m, d))

    cdef long[:, :, :, ::1] S
    S = np.zeros((m, 2**scaling, max_leaves, m), dtype=np.int)

    cdef unsigned int a_it, nb_leaf, va,  counter, ci, cj, ite, pow_set_size, nv, ns, add
    cdef unsigned int b, leaf_numb, i, s, j, i1, i2, l, na_bool, o_all, csi, cs

    cdef vector[vector[long]] C_buff, va_id, buff
    cdef vector[long] coal_va, remove_va, node_id_b, buff_l
    buff.resize(max_leaves)
    for i in range(max_leaves):
        buff[i].resize(1)

    cdef vector[vector[vector[long]]] node_id_v2, C_b, Sm, node_id
    node_id_v2.resize(max_leaves)
    C_b.resize(max_leaves)
    Sm.resize(max_leaves)
    node_id.resize(max_leaves)

    cdef vector[vector[vector[vector[long]]]] node_id_va
    node_id_va.resize(m)
    for i in range(m):
        node_id_va[i].resize(max_leaves)

    cdef int S_size, va_size,  b_it, nv_bool

    cdef long set_size

    if C[0].size() != 0:
        C_buff = C
        for ci in range(C.size()):
            for cj in range(C[ci].size()):
                coal_va.push_back(C[ci][cj])

        for i in range(m):
            if not std_find[vector[long].iterator, long](coal_va.begin(), coal_va.end(), i) != coal_va.end():
                remove_va.push_back(i)
                buff[0][0] = i
                va_id.push_back(buff[0])
            else:
                for ci in range(C_buff.size()):
                    if (std_find[vector[long].iterator, long](C_buff[ci].begin(), C_buff[ci].end(), i) != C_buff[ci].end()):
                        va_id.push_back(C_buff[ci])
                        std_remove[vector[vector[long]].iterator, vector[long]](C_buff.begin(), C_buff.end(), C_buff[ci])
                        C_buff.pop_back()
                        break

    else:
        for i in range(m):
            buff[0][0] = i
            va_id.push_back(buff[0])

    cdef double p_s, p_si, coef, coef_0

    for b in range(n_trees):
        for leaf_numb in range(phi_b.shape[0]):
            for counter in range(phi_b.shape[1]):
                for i in range(N):
                    for j in range(m):
                        for i2 in range(d):
                            phi_b[leaf_numb, counter, i, j, i2] = 0
        nb_leaf = leaves_nb[b]
        for leaf_numb in prange(nb_leaf, nogil=True, schedule='dynamic'):
            node_id_v2[leaf_numb].clear()

            if C[0].size() != 0:
                C_b[leaf_numb] = C
                for nv in range(node_idx_trees[b][leaf_numb].size()):
                    add = 0
                    for ns in range(remove_va.size()):
                        if node_idx_trees[b][leaf_numb][nv] == remove_va[ns]:
                            add = 1
                            buff[leaf_numb][0] = node_idx_trees[b][leaf_numb][nv]
                            node_id_v2[leaf_numb].push_back(buff[leaf_numb])
                            break
                    if add == 0:
                        for ci in range(C_b[leaf_numb].size()):
                            for cj in range(C_b[leaf_numb][ci].size()):
                                if C_b[leaf_numb][ci][cj] == node_idx_trees[b][leaf_numb][nv]:
                                    add = 1
                                    node_id_v2[leaf_numb].push_back(C_b[leaf_numb][ci])
                                    break
                            if add == 1:
                                std_remove[vector[vector[long]].iterator, vector[long]](C_b[leaf_numb].begin(), C_b[leaf_numb].end(), C_b[leaf_numb][ci])
                                C_b[leaf_numb].pop_back()
                                break
                node_id[leaf_numb] = node_id_v2[leaf_numb]
            else:
                node_id[leaf_numb].clear()
                for i in range(node_idx_trees[b][leaf_numb].size()):
                    buff[leaf_numb][0] = node_idx_trees[b][leaf_numb][i]
                    node_id[leaf_numb].push_back(buff[leaf_numb])


            for va in range(va_id.size()):
                node_id_va[va][leaf_numb] = node_id[leaf_numb]
                if not std_find[vector[vector[long]].iterator, vector[long]](node_id_va[va][leaf_numb].begin(), node_id_va[va][leaf_numb].end(), va_id[va]) != node_id_va[va][leaf_numb].end():
                    continue

                std_remove[vector[vector[long]].iterator, vector[long]](node_id_va[va][leaf_numb].begin(), node_id_va[va][leaf_numb].end(), va_id[va])
                node_id_va[va][leaf_numb].pop_back()

                set_size = node_id_va[va][leaf_numb].size()
                pow_set_size = 2**set_size

                for counter in range(0, pow_set_size):
                    va_size = 0
                    S_size = 0
                    for ci in range(set_size):
                        if((counter & (1 << ci)) > 0):
                            for cj in range(node_id_va[va][leaf_numb][ci].size()):
                                S[va, counter, leaf_numb, S_size] = node_id_va[va][leaf_numb][ci][cj]
                                S_size = S_size + 1
                            va_size = va_size + 1

                    for i in range(N):

                        csi = 0
                        cs = 0

                        o_all = 0
                        for s in range(S_size):
                            if ((X[i, S[va, counter, leaf_numb, s]] <= partition_leaves_trees[b, leaf_numb, S[va, counter, leaf_numb, s], 1]) * (X[i, S[va, counter, leaf_numb, s]] > partition_leaves_trees[b, leaf_numb, S[va, counter, leaf_numb, s], 0])):
                                o_all = o_all + 1

                        if o_all == S_size:
                            cs = 1
                            nv_bool = 0
                            for nv in range(va_id[va].size()):
                                if ((X[i, va_id[va][nv]] > partition_leaves_trees[b, leaf_numb, va_id[va][nv], 1]) or (X[i, va_id[va][nv]] <= partition_leaves_trees[b, leaf_numb, va_id[va][nv], 0])):
                                    nv_bool = nv_bool + 1
                                    continue

                            if nv_bool == 0:
                                csi = 1
                        coef = 0
                        for l in range(1, m - set_size):
                            coef = coef + (1.*binomialC(m - set_size - 1, l))/binomialC(m - 1, l + va_size) if binomialC(m - 1, l + va_size) !=0 else 0

                        coef_0 = 1./binomialC(m-1, va_size) if binomialC(m-1, va_size) !=0 else 0

                        if S_size == 0:
                            p_s = lm[leaf_numb]/data.shape[0]
                        else:
                            p_s = (cs * lm[leaf_numb])/lm_s[va, leaf_numb, counter] if lm_s[va, leaf_numb, counter] !=0 else 0
                        p_si = (csi * lm[leaf_numb])/lm_si[va, leaf_numb, counter] if lm_si[va, leaf_numb, counter] !=0 else 0

                        for nv in range(va_id[va].size()):
                            for i2 in range(d):
                                phi_b[leaf_numb, counter, i, va_id[va][nv], i2] += (coef_0 + coef) * (p_si - p_s) * values[b, leaf_idx_trees[b, leaf_numb], i2]

        for i in range(N):
            for j in range(m):
                for i2 in range(d):
                    for leaf_numb in range(phi_b.shape[0]):
                        for counter in range(phi_b.shape[1]):
                            phi[i, j, i2] += phi_b[leaf_numb, counter, i, j, i2]

    return np.asarray(phi)/m


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef global_sdp_reg(double[:, :] X, double[:] fX, double tX,
            double[:] y_pred, double[:, :] data,
            double[:, :, :] values, double[:, :, :, :] partition_leaves_trees,
            long[:, :] leaf_idx_trees, long[:] leaves_nb, double[:] scaling, list C, double pi_level,
            int minimal):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]
    cdef unsigned int max_leaves = partition_leaves_trees.shape[1]
    cdef int n_trees = values.shape[0]
    cdef vector[int].iterator t
    cdef double[:, :, :] leaves_tree
    cdef double[:, :, :] leaves_tree_l
    cdef double[:, :] leaf_part
    cdef double[:, :] leaf_part_l
    cdef double value
    cdef double value_l

    cdef double[:, :] mean_forest
    mean_forest = np.zeros((N, 3))

    cdef double[:, :, :, :, :] mean_forest_b
    mean_forest_b = np.zeros((n_trees, max_leaves, max_leaves, N, 3))

    cdef double[:] sdp
    cdef double[:] sdp_global
    sdp = np.zeros((N))
    sdp_global = np.zeros((m))

    cdef long[:] lm, lm_u, lm_d, lm_s
    cdef unsigned int it, it_s, a_it, b_it, o_all, p, p_s, nb_leaf, p_u, p_d, p_su, p_sd, down, up
    cdef double ss, ss_a, ss_u, ss_d
    cdef int b, leaf_numb, i, s, s_0, s_1, S_size, j, max_size, size, leaf_numb_l

    cdef long[:] S, len_s_star
    len_s_star = np.zeros((N), dtype=np.int)
    cdef double[:, :, :, :] p_n, p_u_n, p_d_n
    p_n = np.zeros((n_trees, max_leaves, max_leaves, N))
    p_u_n = np.zeros((n_trees, max_leaves, max_leaves, N))
    p_d_n = np.zeros((n_trees, max_leaves, max_leaves, N))
    cdef double n, n_u, n_d
    cdef list power, va_id

    cdef vector[long] R, r
    R.resize(N)
    for i in range(N):
        R[i] = i
    r.resize(N)

    cdef long[:] R_buf
    R_buf = np.zeros((N), dtype=np.int)

    cdef double[:] norm, norm_u, norm_d
    norm = np.zeros((n_trees))
    norm_u = np.zeros((n_trees))
    norm_d = np.zeros((n_trees))

    if C[0] != []:
        remove_va = [C[ci][cj] for ci in range(len(C)) for cj in range(len(C[ci]))]
        va_id = [[i] for i in range(m) if i not in remove_va] + C
    else:
        va_id = [[i] for i in range(m)]

    m = len(va_id)
    power = []
    max_size = 0
    for size in range(m + 1):
        power_b = []
        for co in itertools.combinations(va_id, size):
            power_b.append(np.array(sum(list(co),[])))
            max_size += 1
        power.append(power_b)
        if max_size >= 2**15:
            break

    cdef vector[vector[vector[long]]] power_cpp = power
    cdef long[:, :] s_star
    s_star = -1*np.ones((N, X.shape[1]), dtype=np.int)


    cdef long power_set_size = 2**m
    S = np.zeros((data.shape[1]), dtype=np.int)

    for s_0 in tqdm(range(minimal, m + 1)):
        for s_1 in range(0, power_cpp[s_0].size()):
            for i in range(power_cpp[s_0][s_1].size()):
                S[i] = power_cpp[s_0][s_1][i]

            S_size = power_cpp[s_0][s_1].size()
            r.clear()
            N = R.size()
            for i in range(N):
                R_buf[i] = R[i]

            for b in range(n_trees):
                nb_leaf = leaves_nb[b]
                for leaf_numb in range(nb_leaf):
                    for i in range(N):
                        for j in range(3):
                            mean_forest_b[b, leaf_numb, R_buf[i], j] = 0
            for b in range(n_trees):
                for l in range(n_trees):
                    if b == l:
                        leaves_tree = partition_leaves_trees[b]
                        nb_leaf = leaves_nb[b]

                        for leaf_numb in range(nb_leaf):
                            leaf_part = leaves_tree[leaf_numb]
                            value = values[b, leaf_idx_trees[b, leaf_numb], 0]

                            lm = np.zeros(data.shape[0], dtype=np.int)
                            lm_s = np.zeros(data.shape[0], dtype=np.int)

                            for i in prange(data.shape[0], nogil=True):
                                a_it = 0
                                b_it = 0
                                for s in range(m):
                                    if ((data[i, s] <= leaf_part[s, 1]) and (data[i, s] >= leaf_part[s, 0])):
                                        a_it = a_it + 1
                                for s in range(S_size):
                                    if ((data[i, S[s]] <= leaf_part[S[s], 1]) and (data[i, S[s]] >= leaf_part[S[s], 0])):
                                        b_it = b_it + 1

                                if a_it == m:
                                    lm[i] = 1

                                if b_it == S_size:
                                    lm_s[i] = 1

                            for i in prange(N, nogil=True):
                                o_all = 0
                                for s in range(S_size):
                                    if ((X[R_buf[i], S[s]] > leaf_part[S[s], 1]) or (X[R_buf[i], S[s]] < leaf_part[S[s], 0])):
                                        o_all = o_all + 1
                                if o_all > 0:
                                    continue

                                p = 0
                                p_u = 0
                                p_d = 0
                                p_s = 0
                                p_su = 0
                                p_sd = 0

                                for j in range(data.shape[0]):
                                    p += lm[j]
                                    p_s += lm_s[j]
                                    if (fX[R_buf[i]] - y_pred[j])*(fX[R_buf[i]] - y_pred[j]) > tX:
                                        p_u += lm[j]
                                        p_su += lm_s[j]
                                    else:
                                        p_d += lm[j]
                                        p_sd += lm_s[j]

                                mean_forest_b[b, leaf_numb, leaf_numb, R_buf[i], 0] += (p * value*value) / (p_s) - (2 * fX[R_buf[i]] * p * value)/(p_s) if p_s != 0 else 0
                                mean_forest_b[b, leaf_numb, leaf_numb, R_buf[i], 1] += (p_u * value*value) / (p_su) - (2 * fX[R_buf[i]] * p_u * value)/(p_su) if p_su != 0 else 0
                                mean_forest_b[b, leaf_numb, leaf_numb, R_buf[i], 2] += (p_d * value*value) / (p_sd) - (2 * fX[R_buf[i]] * p_d * value)/(p_sd) if p_sd != 0 else 0

                                p_n[b, leaf_numb, leaf_numb, R_buf[i]] += (1.*p)/p_s if p_s != 0 else 0
                                p_u_n[b, leaf_numb, leaf_numb, R_buf[i]] += (1.*p_u)/p_su if p_su != 0 else 0
                                p_d_n[b, leaf_numb, leaf_numb, R_buf[i]] += (1.*p_d)/p_sd if p_sd != 0 else 0
                    else:
                        leaves_tree = partition_leaves_trees[b]
                        nb_leaf = leaves_nb[b]
                        leaves_tree_l = partition_leaves_trees[l]
                        nb_leaf_l = leaves_nb[l]

                        for leaf_numb in range(nb_leaf):
                            for leaf_numb_l in range(nb_leaf_l):

                                leaf_part = leaves_tree[leaf_numb]
                                leaf_part_l = leaves_tree_l[leaf_numb_l]
                                value = values[b, leaf_idx_trees[b, leaf_numb], 0]
                                value_l = values[l, leaf_idx_trees[l, leaf_numb_l], 0]

                                lm = np.zeros(data.shape[0], dtype=np.int)
                                lm_s = np.zeros(data.shape[0], dtype=np.int)

                                for i in prange(data.shape[0], nogil=True):
                                    a_it = 0
                                    b_it = 0
                                    for s in range(m):
                                        if ((data[i, s] <= leaf_part[s, 1]) and (data[i, s] >= leaf_part[s, 0]) and (data[i, s] <= leaf_part_l[s, 1]) and (data[i, s] >= leaf_part_l[s, 0])):
                                            a_it = a_it + 1
                                    for s in range(S_size):
                                        if ((data[i, S[s]] <= leaf_part[S[s], 1]) and (data[i, S[s]] >= leaf_part[S[s], 0]) and (data[i, S[s]] <= leaf_part_l[S[s], 1]) and (data[i, S[s]] >= leaf_part_l[S[s], 0])):
                                            b_it = b_it + 1

                                    if a_it == m:
                                        lm[i] = 1

                                    if b_it == S_size:
                                        lm_s[i] = 1

                                for i in prange(N, nogil=True):
                                    o_all = 0
                                    for s in range(S_size):
                                        if ((X[R_buf[i], S[s]] > leaf_part[S[s], 1]) or (X[R_buf[i], S[s]] < leaf_part[S[s], 0]) or (X[R_buf[i], S[s]] > leaf_part_l[S[s], 1]) or (X[R_buf[i], S[s]] < leaf_part_l[S[s], 0])):
                                            o_all = o_all + 1
                                    if o_all > 0:
                                        continue

                                    p = 0
                                    p_u = 0
                                    p_d = 0
                                    p_s = 0
                                    p_su = 0
                                    p_sd = 0

                                    for j in range(data.shape[0]):
                                        p += lm[j]
                                        p_s += lm_s[j]
                                        if (fX[R_buf[i]] - y_pred[j])*(fX[R_buf[i]] - y_pred[j]) > tX:
                                            p_u += lm[j]
                                            p_su += lm_s[j]
                                        else:
                                            p_d += lm[j]
                                            p_sd += lm_s[j]

                                    mean_forest_b[b, leaf_numb, leaf_numb_l, R_buf[i], 0] += (p * value*value_l) / (p_s)  if p_s != 0 else 0
                                    mean_forest_b[b, leaf_numb, leaf_numb_l, R_buf[i], 1] += (p_u * value*value_l) / (p_su)  if p_su != 0 else 0
                                    mean_forest_b[b, leaf_numb, leaf_numb_l, R_buf[i], 2] += (p_d * value*value_l) / (p_sd) if p_sd != 0 else 0

                                    p_n[b, leaf_numb, leaf_numb_l, R_buf[i]] += (1.*p)/p_s if p_s != 0 else 0
                                    p_u_n[b, leaf_numb, leaf_numb_l, R_buf[i]] += (1.*p_u)/p_su if p_su != 0 else 0
                                    p_d_n[b, leaf_numb, leaf_numb_l, R_buf[i]] += (1.*p_d)/p_sd if p_sd != 0 else 0
            for i in range(N):
                ss_u = 0
                ss_d = 0
                ss_a = 0
                for b in range(n_trees):
                    for l in range(n_trees):
                        if b == l:
                            n = 0
                            n_u = 0
                            n_d = 0
                            nb_leaf = leaves_nb[b]
                            for leaf_numb in range(nb_leaf):
                                n += p_n[b, leaf_numb, leaf_numb, R_buf[i]]
                                n_u += p_u_n[b, leaf_numb, leaf_numb, R_buf[i]]
                                n_d += p_d_n[b, leaf_numb, leaf_numb, R_buf[i]]
                            norm[b] = n
                            norm_u[b] = n_u
                            norm_d[b] = n_d
                            for leaf_numb in range(nb_leaf):
                                ss_a += mean_forest_b[b, leaf_numb, leaf_numb,  R_buf[i], 0]/norm[b] if norm[b] !=0 else 0
                                ss_u += mean_forest_b[b, leaf_numb, leaf_numb, R_buf[i], 1]/norm_u[b] if norm_u[b] !=0 else 0
                                ss_d += mean_forest_b[b, leaf_numb, leaf_numb, R_buf[i], 2]/norm_d[b] if norm_d[b] !=0 else 0
                        else:
                            nb_leaf = leaves_nb[b]
                            nb_leaf_l = leaves_nb[l]

                            for leaf_numb in range(nb_leaf):
                                n = 0
                                n_u = 0
                                n_d = 0
                                for leaf_numb_l in range(nb_leaf_l):
                                    n += p_n[b, leaf_numb, leaf_numb_l, R_buf[i]]
                                    n_u += p_u_n[b, leaf_numb, leaf_numb_l, R_buf[i]]
                                    n_d += p_d_n[b, leaf_numb, leaf_numb_l, R_buf[i]]

                                for leaf_numb_l in range(nb_leaf_l):
                                    ss_a += (p_n[b, leaf_numb, leaf_numb, R_buf[i]]/norm[b])*(mean_forest_b[b, leaf_numb, leaf_numb_l,  R_buf[i], 0]/n) if n*norm[b] !=0  else 0
                                    ss_u += (p_u_n[b, leaf_numb, leaf_numb, R_buf[i]]/norm_u[b])*(mean_forest_b[b, leaf_numb, leaf_numb_l, R_buf[i], 1]/n_u) if n_u*norm_u[b] !=0 else 0
                                    ss_d += (p_d_n[b, leaf_numb, leaf_numb, R_buf[i]]/norm_d[b])*(mean_forest_b[b, leaf_numb, leaf_numb_l, R_buf[i], 2]/n_d) if n_d*norm_d[b] !=0 else 0


                ss = (ss_u - ss_a)/(ss_u - ss_d) if ss_u - ss_d !=0 else 0
                if ss >= sdp[R_buf[i]]:
                    sdp[R_buf[i]] = ss
                    len_s_star[R_buf[i]] = S_size
                    for s in range(S_size):
                        s_star[R_buf[i], s] = S[s]

                if S_size == X.shape[1]:
                    sdp[R_buf[i]] = 1
                    len_s_star[R_buf[i]] = S_size
                    for s in range(S_size):
                        s_star[R_buf[i], s] = S[s]
                    for s in range(len_s_star[R_buf[i]], X.shape[1]): # to filter (important for coalition)
                        s_star[R_buf[i], s] = -1

        for i in range(N):
            if sdp[R_buf[i]] >= pi_level:
                r.push_back(R[i])
                for s in range(len_s_star[R_buf[i]]):
                    sdp_global[s_star[R_buf[i], s]] += 1

        for i in range(r.size()):
            std_remove[vector[long].iterator, long](R.begin(), R.end(), r[i])
            R.pop_back()

        if R.size() == 0 or S_size >= X.shape[1]/2:
            break

    return np.asarray(sdp_global)/X.shape[0], np.array(s_star, dtype=np.long), np.array(len_s_star, dtype=np.long), np.array(sdp)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef global_sdp_reg(double[:, :] X, double[:] fX, double tX,
            double[:] y_pred, double[:, :] data,
            double[:, :, :] values, double[:, :, :, :] partition_leaves_trees,
            long[:, :] leaf_idx_trees, long[:] leaves_nb, double[:] scaling, list C, double pi_level,
            int minimal):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]
    cdef unsigned int max_leaves = partition_leaves_trees.shape[1]
    cdef int n_trees = values.shape[0]
    cdef vector[int].iterator t
    cdef double[:, :, :] leaves_tree
    cdef double[:, :, :] leaves_tree_l
    cdef double[:, :] leaf_part
    cdef double[:, :] leaf_part_l
    cdef double value
    cdef double value_l

    cdef double[:, :] mean_forest
    mean_forest = np.zeros((N, 3))

    cdef double[:, :, :, :, :] mean_forest_b
    mean_forest_b = np.zeros((n_trees, max_leaves, max_leaves, N, 3))

    cdef double[:] sdp, sdp_b
    cdef double[:] sdp_global
    sdp = np.zeros((N))
    sdp_b = np.zeros((N))
    sdp_global = np.zeros((m))

    cdef long[:] lm, lm_u, lm_d, lm_s
    cdef unsigned int it, it_s, a_it, b_it, o_all, p, p_s, nb_leaf, p_u, p_d, p_su, p_sd, down, up
    cdef double ss, ss_a, ss_u, ss_d
    cdef int b, leaf_numb, i, s, s_0, s_1, S_size, j, max_size, size, leaf_numb_l

    cdef long[:] S, len_s_star
    len_s_star = np.zeros((N), dtype=np.int)
    cdef double[:, :, :, :] p_n, p_u_n, p_d_n
    p_n = np.zeros((n_trees, max_leaves, max_leaves, N))
    p_u_n = np.zeros((n_trees, max_leaves, max_leaves, N))
    p_d_n = np.zeros((n_trees, max_leaves, max_leaves, N))
    cdef double n, n_u, n_d
    cdef list power, va_id

    cdef vector[long] R, r
    R.resize(N)
    for i in range(N):
        R[i] = i
    r.resize(N)

    cdef long[:] R_buf
    R_buf = np.zeros((N), dtype=np.int)

    cdef double[:] norm, norm_u, norm_d
    norm = np.zeros((n_trees))
    norm_u = np.zeros((n_trees))
    norm_d = np.zeros((n_trees))

    if C[0] != []:
        remove_va = [C[ci][cj] for ci in range(len(C)) for cj in range(len(C[ci]))]
        va_id = [[i] for i in range(m) if i not in remove_va] + C
    else:
        va_id = [[i] for i in range(m)]

    m = len(va_id)
    power = []
    max_size = 0
    for size in range(m + 1):
        power_b = []
        for co in itertools.combinations(va_id, size):
            power_b.append(np.array(sum(list(co),[])))
            max_size += 1
        power.append(power_b)
        if max_size >= 2**15:
            break

    cdef vector[vector[vector[long]]] power_cpp = power
    cdef long[:, :] s_star
    s_star = -1*np.ones((N, X.shape[1]), dtype=np.int)


    cdef long power_set_size = 2**m
    S = np.zeros((data.shape[1]), dtype=np.int)

    for s_0 in tqdm(range(minimal, m + 1)):
        for s_1 in range(0, power_cpp[s_0].size()):
            for i in range(power_cpp[s_0][s_1].size()):
                S[i] = power_cpp[s_0][s_1][i]

            S_size = power_cpp[s_0][s_1].size()
            r.clear()
            N = R.size()
            for i in range(N):
                R_buf[i] = R[i]

            # sdp_b = compute_sdp_reg(X, fX, tX, y_pred, S[:S_size], data, values, partition_leaves_trees, leaf_idx_trees,
            #            leaves_nb, scaling, 0)

            mean_forest_b = np.zeros((n_trees, max_leaves, max_leaves, N, 3))
            p_n = np.zeros((n_trees, max_leaves, max_leaves, N))
            p_u_n = np.zeros((n_trees, max_leaves, max_leaves, N))
            p_d_n = np.zeros((n_trees, max_leaves, max_leaves, N))

            for b in range(n_trees):
                for l in range(n_trees):
                    if b == l:
                        leaves_tree = partition_leaves_trees[b]
                        nb_leaf = leaves_nb[b]

                        for leaf_numb in range(nb_leaf):
                            leaf_part = leaves_tree[leaf_numb]
                            value = values[b, leaf_idx_trees[b, leaf_numb], 0]

                            lm = np.zeros(data.shape[0], dtype=np.int)
                            lm_s = np.zeros(data.shape[0], dtype=np.int)

                            for i in range(data.shape[0]):
                                a_it = 0
                                b_it = 0
                                for s in range(m):
                                    if ((data[i, s] <= leaf_part[s, 1]) and (data[i, s] >= leaf_part[s, 0])):
                                        a_it = a_it + 1
                                for s in range(S_size):
                                    if ((data[i, S[s]] <= leaf_part[S[s], 1]) and (data[i, S[s]] >= leaf_part[S[s], 0])):
                                        b_it = b_it + 1

                                if a_it == m:
                                    lm[i] = 1

                                if b_it == S_size:
                                    lm_s[i] = 1

                            for i in range(N):
                                o_all = 0
                                for s in range(S_size):
                                    if ((X[i, S[s]] > leaf_part[S[s], 1]) or (X[i, S[s]] < leaf_part[S[s], 0])):
                                        o_all = o_all + 1
                                if o_all > 0:
                                    continue

                                p = 0
                                p_u = 0
                                p_d = 0
                                p_s = 0
                                p_su = 0
                                p_sd = 0

                                for j in range(data.shape[0]):
                                    p += lm[j]
                                    p_s += lm_s[j]
                                    if (fX[i] - y_pred[j])*(fX[i] - y_pred[j]) > tX:
                                        p_u += lm[j]
                                        p_su += lm_s[j]
                                    else:
                                        p_d += lm[j]
                                        p_sd += lm_s[j]

                                mean_forest_b[b, leaf_numb, leaf_numb, i, 0] += (p * value*value) / (p_s) - (2 * fX[i] * p * value)/(p_s) if p_s != 0 else 0
                                mean_forest_b[b, leaf_numb, leaf_numb, i, 1] += (p_u * value*value) / (p_su) - (2 * fX[i] * p_u * value)/(p_su) if p_su != 0 else 0
                                mean_forest_b[b, leaf_numb, leaf_numb, i, 2] += (p_d * value*value) / (p_sd) - (2 * fX[i] * p_d * value)/(p_sd) if p_sd != 0 else 0

                                p_n[b, leaf_numb, leaf_numb, i] += (1.*p)/p_s if p_s != 0 else 0
                                p_u_n[b, leaf_numb, leaf_numb, i] += (1.*p_u)/p_su if p_su != 0 else 0
                                p_d_n[b, leaf_numb, leaf_numb, i] += (1.*p_d)/p_sd if p_sd != 0 else 0
                    else:
                        leaves_tree = partition_leaves_trees[b]
                        nb_leaf = leaves_nb[b]
                        leaves_tree_l = partition_leaves_trees[l]
                        nb_leaf_l = leaves_nb[l]

                        for leaf_numb in range(nb_leaf):
                            for leaf_numb_l in range(nb_leaf_l):

                                leaf_part = leaves_tree[leaf_numb]
                                leaf_part_l = leaves_tree_l[leaf_numb_l]
                                value = values[b, leaf_idx_trees[b, leaf_numb], 0]
                                value_l = values[l, leaf_idx_trees[l, leaf_numb_l], 0]

                                lm = np.zeros(data.shape[0], dtype=np.int)
                                lm_s = np.zeros(data.shape[0], dtype=np.int)

                                for i in range(data.shape[0]):
                                    a_it = 0
                                    b_it = 0
                                    for s in range(m):
                                        if ((data[i, s] <= leaf_part[s, 1]) and (data[i, s] >= leaf_part[s, 0]) and (data[i, s] <= leaf_part_l[s, 1]) and (data[i, s] >= leaf_part_l[s, 0])):
                                            a_it = a_it + 1
                                    for s in range(S_size):
                                        if ((data[i, S[s]] <= leaf_part[S[s], 1]) and (data[i, S[s]] >= leaf_part[S[s], 0]) and (data[i, S[s]] <= leaf_part_l[S[s], 1]) and (data[i, S[s]] >= leaf_part_l[S[s], 0])):
                                            b_it = b_it + 1

                                    if a_it == m:
                                        lm[i] = 1

                                    if b_it == S_size:
                                        lm_s[i] = 1

                                for i in range(N):
                                    o_all = 0
                                    for s in range(S_size):
                                        if ((X[i, S[s]] > leaf_part[S[s], 1]) or (X[i, S[s]] < leaf_part[S[s], 0]) or (X[i, S[s]] > leaf_part_l[S[s], 1]) or (X[i, S[s]] < leaf_part_l[S[s], 0])):
                                            o_all = o_all + 1
                                    if o_all > 0:
                                        continue

                                    p = 0
                                    p_u = 0
                                    p_d = 0
                                    p_s = 0
                                    p_su = 0
                                    p_sd = 0

                                    for j in range(data.shape[0]):
                                        p += lm[j]
                                        p_s += lm_s[j]
                                        if (fX[i] - y_pred[j])*(fX[i] - y_pred[j]) > tX:
                                            p_u += lm[j]
                                            p_su += lm_s[j]
                                        else:
                                            p_d += lm[j]
                                            p_sd += lm_s[j]

                                    mean_forest_b[b, leaf_numb, leaf_numb_l, i, 0] += (p * value*value_l) / (p_s)  if p_s != 0 else 0
                                    mean_forest_b[b, leaf_numb, leaf_numb_l, i, 1] += (p_u * value*value_l) / (p_su)  if p_su != 0 else 0
                                    mean_forest_b[b, leaf_numb, leaf_numb_l, i, 2] += (p_d * value*value_l) / (p_sd) if p_sd != 0 else 0

                                    p_n[b, leaf_numb, leaf_numb_l, i] += (1.*p)/p_s if p_s != 0 else 0
                                    p_u_n[b, leaf_numb, leaf_numb_l, i] += (1.*p_u)/p_su if p_su != 0 else 0
                                    p_d_n[b, leaf_numb, leaf_numb_l, i] += (1.*p_d)/p_sd if p_sd != 0 else 0
            for i in range(N):
                ss_u = 0
                ss_d = 0
                ss_a = 0
                for b in range(n_trees):
                    for l in range(n_trees):
                        if b == l:
                            n = 0
                            n_u = 0
                            n_d = 0
                            nb_leaf = leaves_nb[b]
                            for leaf_numb in range(nb_leaf):
                                n += p_n[b, leaf_numb, leaf_numb, i]
                                n_u += p_u_n[b, leaf_numb, leaf_numb, i]
                                n_d += p_d_n[b, leaf_numb, leaf_numb, i]
                            norm[b] = n
                            norm_u[b] = n_u
                            norm_d[b] = n_d
                            for leaf_numb in range(nb_leaf):
                                ss_a += mean_forest_b[b, leaf_numb, leaf_numb,  i, 0]/norm[b] if norm[b] !=0 else 0
                                ss_u += mean_forest_b[b, leaf_numb, leaf_numb, i, 1]/norm_u[b] if norm_u[b] !=0 else 0
                                ss_d += mean_forest_b[b, leaf_numb, leaf_numb, i, 2]/norm_d[b] if norm_d[b] !=0 else 0
                        else:
                            nb_leaf = leaves_nb[b]
                            nb_leaf_l = leaves_nb[l]

                            for leaf_numb in range(nb_leaf):
                                n = 0
                                n_u = 0
                                n_d = 0
                                for leaf_numb_l in range(nb_leaf_l):
                                    n += p_n[b, leaf_numb, leaf_numb_l, i]
                                    n_u += p_u_n[b, leaf_numb, leaf_numb_l, i]
                                    n_d += p_d_n[b, leaf_numb, leaf_numb_l, i]

                                for leaf_numb_l in range(nb_leaf_l):
                                    ss_a += (p_n[b, leaf_numb, leaf_numb, i]/norm[b])*(mean_forest_b[b, leaf_numb, leaf_numb_l,  i, 0]/n) if n*norm[b] !=0  else 0
                                    ss_u += (p_u_n[b, leaf_numb, leaf_numb, i]/norm_u[b])*(mean_forest_b[b, leaf_numb, leaf_numb_l, i, 1]/n_u) if n_u*norm_u[b] !=0 else 0
                                    ss_d += (p_d_n[b, leaf_numb, leaf_numb, i]/norm_d[b])*(mean_forest_b[b, leaf_numb, leaf_numb_l, i, 2]/n_d) if n_d*norm_d[b] !=0 else 0


                sdp_b[i] = (ss_u - ss_a)/(ss_u - ss_d) if ss_u - ss_d !=0 else 0


            for i in range(N):
                if sdp_b[R_buf[i]] >= sdp[R_buf[i]]:
                    sdp[R_buf[i]] = sdp_b[R_buf[i]]
                    len_s_star[R_buf[i]] = S_size
                    for s in range(S_size):
                        s_star[R_buf[i], s] = S[s]

                if S_size == X.shape[1]:
                    sdp[R_buf[i]] = 1
                    len_s_star[R_buf[i]] = S_size
                    for s in range(S_size):
                        s_star[R_buf[i], s] = S[s]
                    for s in range(len_s_star[R_buf[i]], X.shape[1]): # to filter (important for coalition)
                        s_star[R_buf[i], s] = -1

        for i in range(N):
            if sdp[R_buf[i]] >= pi_level:
                r.push_back(R[i])
                for s in range(len_s_star[R_buf[i]]):
                    sdp_global[s_star[R_buf[i], s]] += 1

        for i in range(r.size()):
            std_remove[vector[long].iterator, long](R.begin(), R.end(), r[i])
            R.pop_back()

        if R.size() == 0 or S_size >= X.shape[1]/2:
            break

    return np.asarray(sdp_global)/X.shape[0], np.array(s_star, dtype=np.long), np.array(len_s_star, dtype=np.long), np.array(sdp)
