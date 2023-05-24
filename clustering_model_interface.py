from pandas import DataFrame, Series
from tqdm import tqdm


class ClusteringModelInterface:
    data: DataFrame = None
    control: Series = None
    admissible_pcas: list[int] = None
    images_out_path: str = None

    def __init__(self, data: DataFrame, control: Series, admissible_pcas: list[int],
                 images_out_path: str = "images") -> None:
        self.data = data
        self.control = control
        self.admissible_pcas = admissible_pcas
        self.images_out_path = images_out_path

    def fit(self, *args, **kwargs):
        pass

    def perform_clustering(self):
        pass

    def get_model_name(self) -> str:
        return self.__class__.__name__

    def get_pca_loop(self) -> tqdm:
        result: tqdm = tqdm(self.admissible_pcas)
        result.set_description(self.get_model_name())
        return result
