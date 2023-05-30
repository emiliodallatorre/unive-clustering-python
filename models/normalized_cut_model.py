from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from models.clustering_model_interface import ClusteringModelInterface
from pandas import DataFrame, Series
from matplotlib import pyplot as plt
from time import time
from tqdm import tqdm

from rand_index import calculate_rand_index


class NormalizedCutModel(ClusteringModelInterface):
    admissible_k: list[int] = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    standardized_x: DataFrame = None

    def __init__(self, data: DataFrame, control: Series, admissible_pcas: list[int]) -> None:
        super().__init__(data, control, admissible_pcas)

        self.standardized_x = StandardScaler().fit_transform(data)

    def perform_clustering(self):
        fig, ax = plt.subplots(5, 10, figsize=(50, 40))
        loop: tqdm = self.get_pca_loop()

        for pca_n in loop:
            pca = PCA(n_components=int(pca_n))
            standardized_x_pca = pca.fit_transform(self.standardized_x)

            for k in self.admissible_k:
                loop.set_postfix_str(f"pca: {pca_n}, k: {k}")

                start = time()

                model: SpectralClustering = SpectralClustering(n_clusters=int(k), assign_labels='kmeans', n_init=1000)
                model.fit(standardized_x_pca)
                y_pred = model.labels_

                ax[self.admissible_pcas.index(pca_n)][self.admissible_k.index(k)].scatter(standardized_x_pca[:, 0],
                                                                                          standardized_x_pca[:, 1],
                                                                                          c=y_pred, s=40,
                                                                                          cmap='tab20')
                ax[self.admissible_pcas.index(pca_n)][self.admissible_k.index(k)].set_title(
                    'pca: ' + str(pca_n) + '\n k: ' + str(k),
                    fontsize=30)
                ax[self.admissible_pcas.index(pca_n)][self.admissible_k.index(k)].set_xticks([])
                ax[self.admissible_pcas.index(pca_n)][self.admissible_k.index(k)].set_yticks([])
                ax[self.admissible_pcas.index(pca_n)][self.admissible_k.index(k)].text(0.99, 0.01,
                                                                                       't:' + str(
                                                                                           round(time() - start, 3)),
                                                                                       fontsize=20,
                                                                                       transform=ax[
                                                                                           self.admissible_pcas.index(
                                                                                               pca_n)][
                                                                                           self.admissible_k.index(
                                                                                               k)].transAxes,
                                                                                       verticalalignment='bottom',
                                                                                       horizontalalignment='right',
                                                                                       color='black',
                                                                                       bbox=dict(facecolor='white',
                                                                                                 alpha=0.5))

                rand_index = calculate_rand_index(self.control, y_pred)
                ax[self.admissible_pcas.index(pca_n)][self.admissible_k.index(k)].text(0.01, 0.01,
                                                                                       'ri:' + str(
                                                                                           round(rand_index, 3)),
                                                                                       fontsize=20,
                                                                                       transform=ax[
                                                                                           self.admissible_pcas.index(
                                                                                               pca_n)][
                                                                                           self.admissible_k.index(
                                                                                               k)].transAxes,
                                                                                       verticalalignment='bottom',
                                                                                       horizontalalignment='left',
                                                                                       color='black',
                                                                                       bbox=dict(facecolor='white',
                                                                                                 alpha=0.5))
        plt.tight_layout()
        plt.savefig(f'{self.images_out_path}/{self.get_model_name()}.png')
        plt.show()
