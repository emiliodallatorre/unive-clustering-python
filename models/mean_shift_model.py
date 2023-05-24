from itertools import cycle
from time import time

from matplotlib import pyplot as plt
from numpy import unique
from pandas import DataFrame
from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA
from tqdm import tqdm

from clustering_model_interface import ClusteringModelInterface
from random_index import calculate_random_index


class MeanShiftModel(ClusteringModelInterface):
    admissible_bandwidths: list[float] = [x * 0.25 + 1 for x in range(6)]

    transformed_data: DataFrame = None

    def perform_clustering(self):
        fig, ax = plt.subplots(6, 8, figsize=(50, 40))
        loop: tqdm = self.get_pca_loop()

        for pca_index, pca_n in enumerate(loop):
            pca = PCA(n_components=pca_n)
            pca.fit(self.data)

            self.transformed_data = pca.transform(self.data)

            for bandwidth_index, bandwidth in enumerate(self.admissible_bandwidths):
                loop.set_postfix_str(f"pca: {pca_n}, bandwidth: {bandwidth}")

                start: float = time()
                model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                model.fit(self.transformed_data)
                end: float = time()
                labels = model.labels_
                cluster_centers = model.cluster_centers_
                labels_unique = unique(labels)
                n_clusters_ = len(labels_unique)

                colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
                for k, col in zip(range(n_clusters_), colors):
                    my_members = labels == k
                    cluster_center = cluster_centers[k]
                    ax[bandwidth_index, pca_index].plot(self.transformed_data[my_members, 0], self.transformed_data[my_members, 1],
                                      col + '.')
                    ax[bandwidth_index, pca_index].plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                                      markeredgecolor='k', markersize=14)
                    ax[bandwidth_index, pca_index].set_title(
                        'n_components = %d,\n bandwidth = %.3f' % (pca_n, bandwidth),
                        fontsize=20)
                    ax[bandwidth_index, pca_index].set_xticks(())
                    ax[bandwidth_index, pca_index].set_yticks(())
                    ax[bandwidth_index, pca_index].text(0.99, 0.01, f't = {round(end - start, 2)} s',  # tempo di esecuzione
                                      transform=ax[bandwidth_index, pca_index].transAxes, size=12,
                                      horizontalalignment='right',
                                      verticalalignment='bottom',
                                      color='black',
                                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
                    ax[bandwidth_index, pca_index].text(0.01, 0.01, f'clusters = {n_clusters_}',
                                      transform=ax[bandwidth_index, pca_index].transAxes, size=12,
                                      horizontalalignment='left',
                                      verticalalignment='bottom',
                                      color='black',
                                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

                    random_index = calculate_random_index(self.control, labels)

                    ax[bandwidth_index, pca_index].text(0.99, 0.99, f'rand index = {round(random_index, 2)}',
                                      transform=ax[bandwidth_index, pca_index].transAxes, size=12,
                                      horizontalalignment='right',
                                      verticalalignment='top',
                                      color='black',
                                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        plt.tight_layout()
        plt.savefig(f'{self.images_out_path}/{self.get_model_name()}.png')
        plt.show()
