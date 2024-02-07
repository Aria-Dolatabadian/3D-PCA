import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Read genotypic data from CSV
genotypic_df = pd.read_csv('genotypic_data.csv')

# Perform PCA
pca = PCA(n_components=3)
pca_result = pca.fit_transform(genotypic_df)

# Perform clustering (KMeans)
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)  # Explicitly set n_init to suppress FutureWarning
cluster_labels = kmeans.fit_predict(pca_result)

# Plot 3D PCA with different colors for each cluster
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = ['r', 'g', 'b']  # You can add more colors if you have more clusters
for i in range(num_clusters):
    cluster_indices = np.where(cluster_labels == i)
    ax.scatter(pca_result[cluster_indices, 0], pca_result[cluster_indices, 1], pca_result[cluster_indices, 2], c=colors[i], label=f'Cluster {i+1}')

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('3D PCA Plot with Clusters')
ax.legend()
plt.show()
