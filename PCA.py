import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

# massive note
# you should usually accept principal components until you get a total of 95% explained variance, so you will need to accept 4 principal components.

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Subplot 1 for explained variance ratio
plt.subplot(1, 2, 1)
plt.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_, align='center', color='#66C2A5', label='Individual Explained Variance')
plt.step(range(len(cumulative_explained_variance)), cumulative_explained_variance, where='mid', color='#238B45', label='Cumulative Explained Variance')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.title('Explained Variance Ratio by Principal Component')
plt.legend(loc='best')
plt.xlim(-1, 8.5)

# Subplot 2 for scree plot
plt.subplot(1, 2, 2)
plt.plot(pca.explained_variance_, 'ro-', linewidth=2, color='#D53E4F')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.ylabel('Eigenvalue')
plt.xlim(-1, 8.5)

plt.tight_layout()
plt.show()

# export PCA loadings
loadings_df = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(pca.n_components_)], index=data.columns)
loadings_df.to_csv('pcaloadings.csv')

# export explained variance ratio as a CSV file
explained_variance_ratio_df = pd.DataFrame({'Principal Component': [f'PC{i+1}' for i in range(pca.n_components_)], 'Explained Variance Ratio': pca.explained_variance_ratio_})
explained_variance_ratio_df.to_csv('pcavarianceratio.csv', index=False)