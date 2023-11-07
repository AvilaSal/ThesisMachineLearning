import numpy as np
import pandas as pd

# generate a random value based on conditions
# Function to generate a random value based on your conditions
def generate_random_value():
    prob_1_to_50 = 0.75

    # Randomly decide whether to pick from the first range (1-50) or the second range (51-100)
    if np.random.rand() < prob_1_to_50:
        return np.random.randint(1, 51)
    else:
        return np.random.randint(51, 101)

# Read the data from .csv file
data = pd.read_csv('imputedfull.csv')

# Replace numeric values with random ones
for col in data.columns:
    if data[col].dtype in [np.int64, np.float64]:  # Checks if column is numeric
        data[col] = [generate_random_value() for _ in range(len(data[col]))]

# Save the modified data to a new .csv file
data.to_csv('randomfulldata.csv', index=False)
