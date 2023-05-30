from pandas import DataFrame, read_csv, Series

from models.clustering_model_interface import ClusteringModelInterface
from plots.plot_pca_analysis import PlotPcaAnalysis

data: DataFrame = read_csv('data/semeion.csv', sep=' ', usecols=range(256), names=range(256))
control: DataFrame = read_csv('data/semeion.csv', sep=' ', usecols=range(256, 266), names=range(10))
control: Series = control.idxmax(axis=1)

models: list[ClusteringModelInterface] = [
    # NormalizedCutModel(data, control, [2, 3, 4, 5, 6]),
    # MeanShiftModel(data, control, [2, 3, 4, 5, 6, 8, 10, 20]),
    # GaussianMixtureModel(data, control, [3, 10, 15, 20, 25, 50, 256]),
]

plots: list = [
    # PlotGaussianMixture(data, control),
    PlotPcaAnalysis(data),
]

for model in models:
    model.perform_clustering()

for plot in plots:
    plot.plot()
