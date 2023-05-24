from pandas import DataFrame


class ClusteringModelInterface:
    data: DataFrame = None

    def __init__(self, data: DataFrame) -> None:
        self.data = data

    def fit(self, *args, **kwargs):
        pass

    def cluster(self, pca_components: int = 5) -> DataFrame:
        pass

    def plot_result(self):
        pass

    def save_plot(self, path: str):
        pass
