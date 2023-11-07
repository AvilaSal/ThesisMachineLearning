import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load your data
file_path = "DataTest2.csv"  # Replace with your CSV file path
data = pd.read_csv(file_path)

# If your CSV has an index or labels, adjust the data loading accordingly
# e.g., data = pd.read_csv(file_path, index_col=0) if the first column is an index

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply UMAP
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(data_scaled)

# Plot the results
plt.scatter(embedding[:, 0], embedding[:, 1], s=5)
plt.title('UMAP Projection')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.show()


umap_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
output_path = "umap_results.csv"
umap_df.to_csv(output_path, index=False)

print(f"UMAP results saved to {output_path}")
