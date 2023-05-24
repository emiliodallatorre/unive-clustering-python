import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# load data
df = pd.read_csv('data/semeion.csv', sep=' ', usecols=range(0, 256), names=range(0, 256))
df.head()

# standardizzo i dati ad aver una media di zero e varianza 1
X_std = StandardScaler().fit_transform(df)

# e creo una istanza PCA
fig, ax = plt.subplots(2, 3, figsize=(40, 25))
pca_test: list = [4, 20, 40, 80, 160, 200]
for i in range(6):
    pca = PCA(pca_test[i], random_state=1, svd_solver='full', whiten=True)
    X_pca = pca.fit_transform(X_std)
    ax[i // 3, i % 3].bar(range(1, pca.n_components_ + 1), pca.explained_variance_ratio_, alpha=0.5, align='center')
    ax[i // 3, i % 3].set_title('number of component: = ' + str(pca_test[i]), fontsize=40)
    ax[i // 3, i % 3].set_xlabel('component', fontsize=35, labelpad=20)
    ax[i // 3, i % 3].set_ylabel('explained variance', fontsize=35, labelpad=20)
    ax[i // 3, i % 3].tick_params(axis='both', which='major', labelsize=10)
    ax[i // 3, i % 3].grid(linestyle='-.', linewidth=1, alpha=0.5, which='both', color='black', zorder=0)
plt.tight_layout()
plt.savefig("images/pcaReduction.png")
plt.show()
