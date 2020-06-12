'''
This module handles matrix composition/decomposition for the algorithm.
'''
import numpy as np
import pprint as p
import logging
from random import random
from scipy.sparse import csr_matrix, lil_matrix, csc_matrix
from math import sqrt
import scipy

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

def create_dense_Q(Q_blk, q=None, f=None, d=1):
    """
    This function is used to create a dense Q matrix like:
        [ Q_blk | q ]
    Q = |-----------]
        [   f   | d ]

    Args:
        Q_blk:  A dense matrix.
        q:      A dense matrix. Default: None (a zero-column)
        f:      A dense matrix. Default: None (a zero-line)
        d:      A number. Default: 1

    Returns:
        Q - The computed dense matrix
    """
    m, n = Q_blk.shape
    Q = np.zeros(tuple(map(lambda x: x + 1, Q_blk.shape)))
    Q[0:-1, 0:-1] = Q_blk
    Q[-1, -1] = d

    if q is not None:
        Q[0: -1, -1] = np.reshape(q, m)
    if f is not None:
        Q[-1, 0: -1] = np.reshape(f, n)

    return Q

def create_sparse_Q(Q_blk, q=None, f=None, d=1):
    """
    This function is used to create a sparse Q matrix like:
        [ Q_blk | q ]
    Q = |-----------]
        [   f   | d ]

    Args:
        Q_blk:  A sparse matrix.
        q:      A sparse matrix. Default: None (a zero-column)
        f:      A sparse matrix. Default: None (a zero-line)
        d:      A number. Default: 1

    Returns:
        Q - The computed sparse matrix
    """
    m, n = Q_blk.shape
    Q = csr_matrix((m + 1, n + 1))
    Q[0:-1, 0:-1] = Q_blk
    Q[-1, -1] = d

    if q is not None:
        Q[0: -1, -1] = np.reshape(q, (m, 1))
    if f is not None:
        Q[-1, 0: -1] = np.reshape(f, (1, n))

    return Q

def get_Q_dense_blocks(Q):
    """
    This function extracts from a Q matrix the following blocks:
        Q_blk = Q[0:-1, 0:-1]
        f = Q[-1, 0: -1]
        q = Q[0: -1, -1]
        d = Q[-1, -1]

    Args:
        Q:  A dense matrix

    Returns:
        A tuple: (Q_blk, q, f, d)
    """
    m, n = tuple(map(lambda x: x - 1, Q.shape))
    Q_blk = Q[0:-1, 0:-1]
    f = np.reshape(Q[-1, 0: -1], (1, n))
    q = np.reshape(Q[0: -1, -1], (m, 1))
    d = Q[-1, -1]

    return Q_blk, q, f, d

def get_Q_sparse_blocks(Q):
    """
    This function extracts from a Q matrix the following blocks:
        Q_blk = Q[0:-1, 0:-1]
        f = Q[-1, 0: -1]
        q = Q[0: -1, -1]
        d = Q[-1, -1]

    Args:
        Q:  A sparse matrix

    Returns:
        A tuple: (Q_blk, q, f, d)
    """
    Q_blk = Q[0:-1, 0:-1]
    f = Q[-1, 0: -1]
    q = Q[0: -1, -1]
    d = Q[-1, -1]

    return Q_blk, q, f, d

def get_L(mat_list):
    """
    This function computes the list of Largest Magnitude eigenvalues of the
    product mat.T * mat for given list of matrices 'mat_list'.

    Args:
        mat_list:   The list of matrices (ndarray or sparse).

    Returns:
        list
    """
    L = []
    for i, mat in enumerate(mat_list):
        eig_val = scipy.sparse.linalg.eigs(mat.dot(mat.T), k=1, which='LM',
                                           return_eigenvectors=False)[0]

        L.append(eig_val.real)

    return L

def sparse_vec_norm2(v, ord=2):
    """
    This function computes the second order norm for a sparse vector.

    Args:
        v:  The vector

    Returns:
        float
    """
    return sqrt((v.data ** 2).sum())
