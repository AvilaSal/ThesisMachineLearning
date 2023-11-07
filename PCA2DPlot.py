import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Read the data from .csv file
data = pd.read_csv('datatest2.csv')

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Apply PCA
pca = PCA(n_components=2)  # Change n_components as needed
pca_results = pca.fit_transform(scaled_data)

# export
# df_pca_results = pd.DataFrame(data=pca_results, columns=['Principal Component 1', 'Principal Component 2'])
# df_pca_results.to_csv('pcaresults.csv', index=False)

# Visualize the PCA results
plt.figure(figsize=(8, 6))
plt.scatter(pca_results[:, 0], pca_results[:, 1], edgecolor='k', s=50)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2 Component PCA')
plt.show()
