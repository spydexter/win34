from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Iris dataset
datasets = load_iris()
x = pd.DataFrame(datasets.data, columns=['Sepal_length', 'Sepal_width', 'Petal_length', 'Petal_width'])
y = pd.DataFrame(datasets.target, columns=['Targets'])

# Define color map
colourmap = np.array(['red', 'green', 'blue'])

# Plot the real data
plt.figure(figsize=(14, 7))
plt.subplot(1, 3, 1)
plt.scatter(x.Petal_length, x.Petal_width, c=colourmap[y.Targets], s=40)
plt.title("Real")

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(x)
predy = np.choose(kmeans.labels_, [0, 1, 2]).astype(np.int64)

# Plot KMeans results
plt.subplot(1, 3, 2)
plt.scatter(x.Petal_length, x.Petal_width, c=colourmap[predy], s=40)
plt.title('KMeans')

# Standardize the data
scaler = preprocessing.StandardScaler()
scaler.fit(x)
xsa = scaler.transform(x)
xs = pd.DataFrame(xsa, columns=x.columns)

# Apply Gaussian Mixture Model clustering
gmm = GaussianMixture(n_components=3)
gmm.fit(xs)
y_cluster_gmm = gmm.predict(xs)

# Plot GMM results
plt.subplot(1, 3, 3)
plt.scatter(x.Petal_length, x.Petal_width, c=colourmap[y_cluster_gmm], s=40)
plt.title('GMM Classification')

plt.tight_layout()
plt.show()
