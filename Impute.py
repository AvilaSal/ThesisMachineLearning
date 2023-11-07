import pandas as pd

data = pd.read_csv('3 - Attention.csv')

# data_imputed = data.fillna(data.mean()).round(4)
data_imputed = data.dropna() # to blank rows

data_imputed.to_csv('3 - Attention.csv', index=False)
