import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid

# Load your dataset (replace 'DataTest2.csv' with the actual file path)
data = pd.read_csv('DataTest2.csv')

# Select all columns/features for clustering (exclude any non-feature columns if necessary)
X = data

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce dimensionality
pca = PCA(n_components=4)  # Specify the desired number of components (e.g., 2)
X_pca = pca.fit_transform(X_scaled)

# Define a range of values for epsilon (eps) and minimum samples (min_samples)
param_grid = {
    'eps': [0.1, 0.5, 1.0, 1.5, 2.0],  # Adjust the values as needed
    'min_samples': [2, 3, 4, 5, 6]    # Adjust the values as needed
}

best_score = -1  # Initialize the best silhouette score
best_params = None  # Initialize the best parameter combination

# Perform grid search
for params in ParameterGrid(param_grid):
    dbscan = DBSCAN(**params)
    labels = dbscan.fit_predict(X_pca)

    # Check if DBSCAN clustered the data into more than one cluster
    if len(set(labels)) > 1:
        silhouette_avg = silhouette_score(X_pca, labels)
        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_params = params

print("Best DBSCAN Parameters:", best_params)
print("Best Silhouette Score:", best_score)
