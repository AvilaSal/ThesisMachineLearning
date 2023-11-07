import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Read the data from .csv file
data = pd.read_csv('DataTest2.csv')

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Apply PCA
pca = PCA()
pca_results = pca.fit(scaled_data)

# Cumulative explained variance
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Decide the number of principal components to use
num_components = 4 # Based on the earlier step of cumulative explained variance
pca = PCA(n_components=num_components)
pca_data = pca.fit_transform(scaled_data)

# Manually setting initial values for DBSCAN parameters
initial_eps = 0.5
initial_min_samples = 5

# Fit DBSCAN with the manually chosen parameters
db_manual = DBSCAN(eps=initial_eps, min_samples=initial_min_samples)
labels_manual = db_manual.fit_predict(pca_data)

# Number of clusters in labels, ignoring noise if present
n_clusters_manual = len(set(labels_manual)) - (1 if -1 in labels_manual else 0)
n_noise_manual = list(labels_manual).count(-1)

# Add the cluster labels to the original data
data_with_clusters = data.copy()
data_with_clusters['Cluster'] = labels_manual

# Export the data with clusters to a new CSV file
data_with_clusters.to_csv('clustered_DataTest2.csv', index=False)

# Prepare a DataFrame to hold all descriptive statistics
all_stats = pd.DataFrame()

# Export descriptive statistics for each cluster, including noise
for cluster_label in set(labels_manual):
    cluster_data = data_with_clusters[data_with_clusters['Cluster'] == cluster_label]
    stats = cluster_data.describe().transpose()
    stats['Cluster'] = f'Cluster {cluster_label}' if cluster_label != -1 else 'Noise'
    all_stats = pd.concat([all_stats, stats])

# Export all descriptive statistics to a single CSV file
all_stats.to_csv('ClusterDescriptives.csv', index=True)


# Visualization of the clustering result (only for 2D plots)
# Note: This block is only applicable if you are using a Jupyter Notebook or a similar environment.
plt.figure(figsize=(10, 7))
unique_labels_manual = set(labels_manual)
colors_manual = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels_manual))]

# Create a legend for the clusters
for k, col in zip(unique_labels_manual, colors_manual):
    if k == -1:
        # Black used for noise
        col = [0, 0, 0, 1]
        label = 'Noise'
    else:
        label = f'Cluster {k}'

    class_member_mask = (labels_manual == k)
    xy = pca_data[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=10, label=label)

plt.title('DBSCAN clustering on PCA-reduced data (Manual Parameters)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(loc='best')
plt.show()

from sklearn.metrics import silhouette_samples, silhouette_score

# Compute silhouette scores for each sample
silhouette_vals = silhouette_samples(pca_data, labels_manual)

# Create a DataFrame to hold the silhouette scores for each sample
silhouette_df = pd.DataFrame({
    'Silhouette Score': silhouette_vals,
    'Cluster': labels_manual
})

# Calculate and print the mean silhouette score for each cluster
cluster_labels = np.unique(labels_manual)
for label in cluster_labels:
    if label != -1:  # Exclude noise points
        label_silhouette = silhouette_df[silhouette_df['Cluster'] == label]['Silhouette Score'].mean()
        print(f'Cluster {label} Silhouette Score: {label_silhouette:.2f}')

# Compute the overall mean silhouette score for the dataset, excluding noise
overall_silhouette_score = silhouette_score(pca_data[labels_manual != -1], labels_manual[labels_manual != -1])
print(f'Overall mean Silhouette Score (excluding noise): {overall_silhouette_score:.2f}')


# Print the number of points labeled as noise
n_noise = (labels_manual == -1).sum()
print(f'Number of points labeled as noise: {n_noise}')