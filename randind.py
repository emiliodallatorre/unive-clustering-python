from time import time

from scipy.special import comb


def rand_index_(clustering):
    print("Proviamo a capire...")

    start_time = time()

    # Calculate A
    a = 0
    for cluster in clustering:
        classes_in_cluster: set = set(clustering[cluster])
        for class_ in classes_in_cluster:
            a += comb(clustering[cluster].count(class_), 2)

    # Calculate B
    b = 0

    clusters: list = list(clustering.keys())

    for cluster in clusters:
        for class_ in clustering[cluster]:
            different_classes_outside_cluster: int = 0
            for other_cluster in clusters[clusters.index(cluster) + 1:]:
                b += len(clustering[other_cluster]) - clustering[
                    other_cluster].count(class_)

    # Calculate N
    n = sum([len(clustering[cluster]) for cluster in clustering])

    print(a, b)

    end_time = time()
    print("Tempo di esecuzione: ", end_time - start_time)

    return 2 * (a + b) / (n * (n - 1))


if __name__ == '__main__':
    clusters: dict = {
        0: [0, 0, 1],
        1: [1, 1, 2],
        2: [2, 2],
    }

    print(rand_index_(clusters))
