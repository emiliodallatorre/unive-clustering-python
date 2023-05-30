import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PlotPcaAnalysis:
    images_out_path: str = "images"
    admissible_pcas: list[int] = [4, 20, 40, 80, 160, 200]

    data: DataFrame = None
    standardized_x: DataFrame = None

    def __init__(self, data: DataFrame):
        self.data = data

        self.standardized_x = StandardScaler().fit_transform(data)

    def plot(self):
        fig, ax = plt.subplots(2, 3, figsize=(40, 25))

        for i, pca_n in enumerate(self.admissible_pcas):
            pca = PCA(pca_n, random_state=1, svd_solver='full', whiten=True)

            ax[i // 3, i % 3].bar(range(1, pca.n_components_ + 1), pca.explained_variance_ratio_, alpha=0.5,
                                  align='center')
            ax[i // 3, i % 3].set_title('number of component: = ' + str(pca_n), fontsize=40)
            ax[i // 3, i % 3].set_xlabel('component', fontsize=35, labelpad=20)
            ax[i // 3, i % 3].set_ylabel('explained variance', fontsize=35, labelpad=20)
            ax[i // 3, i % 3].tick_params(axis='both', which='major', labelsize=10)
            ax[i // 3, i % 3].grid(linestyle='-.', linewidth=1, alpha=0.5, which='both', color='black', zorder=0)

        plt.tight_layout()
        plt.savefig(f'{self.images_out_path}/{self.get_plot_name()}.png')

    def get_plot_name(self) -> str:
        return self.__class__.__name__
