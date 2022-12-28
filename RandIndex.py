import numpy as np


def rand_index(clusters, classes):
    """Calculate Rand index.
    Parameters
    ----------
    clusters : array, shape = [n_samples]
        Predicted cluster labels.
    classes : array, shape = [n_samples]
        True class labels.
    Returns
    -------
    ARI : float
        Rand Index
    """

    # Calculate Rand Index
    A = np.c_[(clusters, classes)]
    n_ac = sum([comb2(n_ij) for n_ij in np.bincount(np.sum(A, axis=1))])
    n_bd = sum([comb2(n_ij) for n_ij in np.bincount(np.sum(A, axis=0))])
    sum_comb = sum([comb2(n_ij) for n_ij in np.bincount(np.ravel(A))])
    n = sum_comb * 2
    return (n_ac + n_bd) / n


def comb2(n):
    return n * (n - 1) / 2
