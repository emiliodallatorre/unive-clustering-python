import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class PlotGaussianMixture:
    images_out_path: str = "images"

    data: DataFrame = None
    control: Series = None
    standardized_x: DataFrame = None

    def __init__(self, data: DataFrame, control: Series) -> None:
        self.data = data
        self.control = control

        self.standardized_x = StandardScaler().fit_transform(data)

    def plot(self):
        pca = PCA(n_components=25)
        x_pca = pca.fit_transform(self.standardized_x)

        model: GaussianMixture = GaussianMixture(n_components=12, covariance_type='diag', random_state=1)
        model.fit(x_pca)
        predicted_y = model.predict(x_pca)

        reconstructed_x = pca.inverse_transform(x_pca)

        fig, ax = plt.subplots(3, 5, figsize=(15, 11))
        for i in range(15):
            ax[i // 5, i % 5].imshow(reconstructed_x[predicted_y == i].mean(axis=0).reshape(16, 16), cmap='Blues')
            ax[i // 5, i % 5].set_title('pca: ' + str(pca.n_components_) + '\ncluster n: ' + str(i), fontsize=15)
            ax[i // 5, i % 5].axis('off')

        plt.tight_layout()
        # plt.show()
        plt.savefig(f'{self.images_out_path}/{self.get_plot_name()}.png')

    def get_plot_name(self) -> str:
        return self.__class__.__name__
