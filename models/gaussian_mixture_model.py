from time import time

from matplotlib import pyplot as plt
from pandas import DataFrame, Series
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from models.clustering_model_interface import ClusteringModelInterface
from random_index import calculate_random_index


class GaussianMixtureModel(ClusteringModelInterface):
    admissible_k: list[int] = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    standardized_x: DataFrame = None

    def __init__(self, data: DataFrame, control: Series, admissible_pcas: list[int]) -> None:
        super().__init__(data, control, admissible_pcas)

        self.standardized_x = StandardScaler().fit_transform(data)

    def perform_clustering(self):
        fig, ax = plt.subplots(7, 10, figsize=(50, 40))
        loop: tqdm = self.get_pca_loop()

        for pca_index, pca_n in enumerate(loop):
            pca = PCA(n_components=int(pca_n))
            x_pca = pca.fit_transform(self.standardized_x)

            for k in self.admissible_k:
                loop.set_postfix_str(f"pca: {pca_n}, k: {k}")

                start: float = time()
                model: GaussianMixture = GaussianMixture(n_components=int(k),
                                                      covariance_type='diag', random_state=1)
                model.fit(x_pca)
                y_pred = model.predict(x_pca)

                ax[self.admissible_pcas.index(pca_n)][self.admissible_k.index(k)].scatter(x_pca[:, 0], x_pca[:, 1],
                                                                                          c=y_pred,
                                                                                          s=40,
                                                                                          cmap='tab20')
                ax[self.admissible_pcas.index(pca_n)][self.admissible_k.index(k)].set_title(
                    'pca: ' + str(pca_n) + '\nk: ' + str(k),
                    fontsize=30)
                ax[self.admissible_pcas.index(pca_n)][self.admissible_k.index(k)].set_xticks([])
                ax[self.admissible_pcas.index(pca_n)][self.admissible_k.index(k)].set_yticks([])

                ax[self.admissible_pcas.index(pca_n)][self.admissible_k.index(k)].text(0.99, 0.01,
                                                                                       't:' + str(
                                                                                           round(time() - start, 3)),
                                                                                       fontsize=20,
                                                                                       transform=
                                                                                       ax[self.admissible_pcas.index(
                                                                                           pca_n)][
                                                                                           self.admissible_k.index(
                                                                                               k)].transAxes,
                                                                                       verticalalignment='bottom',
                                                                                       horizontalalignment='right',
                                                                                       color='black',
                                                                                       bbox=dict(facecolor='white',
                                                                                                 alpha=0.5))

                random_index = calculate_random_index(self.control, y_pred)

                ax[self.admissible_pcas.index(pca_n)][self.admissible_k.index(k)].text(0.01, 0.01,
                                                                                       'ri:' + str(round(random_index, 3)),
                                                                                       fontsize=20,
                                                                                       transform=
                                                                                       ax[self.admissible_pcas.index(
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
