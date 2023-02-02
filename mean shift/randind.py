from time import time

from scipy.special import comb


def rand_index_(target, labels):
    raw_clustering: dict = {}
    for i, label in enumerate(labels):
        if label not in raw_clustering:
            raw_clustering[label] = [target[i]]
        else:
            raw_clustering[label].append(target[i])

    start_time = time()

    # Calculate A
    a = 0
    for cluster in raw_clustering:
        classes_in_cluster: set = set(raw_clustering[cluster])
        for class_ in classes_in_cluster:
            a += comb(raw_clustering[cluster].count(class_), 2)

    # Calculate B
    b = 0
    clustering: list = list(raw_clustering.keys())

    for cluster in clustering:
        for class_ in raw_clustering[cluster]:
            for other_cluster in clustering[clustering.index(cluster) + 1:]:
                b += len(raw_clustering[other_cluster]) - raw_clustering[
                    other_cluster].count(class_)

    # Calculate N
    n = sum([len(raw_clustering[cluster]) for cluster in raw_clustering])

    # print(a, b)

    end_time = time()
    # rint("Tempo di esecuzione: ", end_time - start_time)

    return 2 * (a + b) / (n * (n - 1))
