import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the clustered data
data = pd.read_csv('clustered_DataTest2.csv')

# Filter out the noise (label -1)
data_no_noise = data[data['Cluster'] != -1]

# Prepare the data for classification
X = data_no_noise.drop('Cluster', axis=1)  # Features
y = data_no_noise['Cluster']  # Target variable

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initialize the Random Forest classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_clf.fit(X_train, y_train)

# Make predictions
y_pred = rf_clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred, labels=np.unique(y_pred)))

# Plot feature importances
feature_importances = rf_clf.feature_importances_
features = data.columns[:-1]  # Exclude the 'Cluster' column
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features)
plt.title('Feature Importances')
plt.show()

# Get unique labels (cluster numbers) from the true labels and predictions
unique_labels = np.unique(np.concatenate((y_test, y_pred)))

# Create the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=unique_labels)

# Create a DataFrame from the confusion matrix and label the indices and columns
conf_matrix_df = pd.DataFrame(conf_matrix,
                              index=['Cluster {}'.format(i) for i in unique_labels],
                              columns=['Cluster {}'.format(i) for i in unique_labels])

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_df, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Generate a classification report
report = classification_report(y_test, y_pred, output_dict=True)

# Turn the classification report into a DataFrame
report_df = pd.DataFrame(report).transpose().reset_index()
report_df.rename({'index': 'Metric'}, axis=1, inplace=True)

# Get feature importances
feature_importances = rf_clf.feature_importances_
feature_names = data.columns[:-1]  # Exclude the 'Cluster' column

# Create a DataFrame for feature importances
importances_df = pd.DataFrame({'Metric': feature_names, 'Score': feature_importances})

# Add a row for accuracy in the feature importances DataFrame
accuracy_df = pd.DataFrame({'Metric': ['Accuracy'], 'Score': [accuracy]})

# Combine the importances DataFrame with the report DataFrame
combined_df = pd.concat([importances_df, accuracy_df, report_df], ignore_index=True)

# Export the combined DataFrame to a CSV file
combined_df.to_csv('model_performance_and_feature_importances.csv', index=False)

first_tree = rf_clf.estimators_[0]  # Access the first decision tree


from sklearn.tree import plot_tree

# List of feature names as strings
feature_names = ['Attention', 'Speed', 'Fluency', 'Mood', 'Intelligence', 'visuomotor']

# Specify the index of the tree you want to plot
# python uses zero-based indexing. For example, the third tree has an index of 2. The range of trees is 0-99 if I set 100 trees as my parameter.
tree_index_to_plot = 1  # Change this to the index of the tree you want to visualize

# Access the selected decision tree from the Random Forest
selected_tree = rf_clf.estimators_[tree_index_to_plot]

# Plot the selected decision tree
plt.figure(figsize=(12, 8))
plot_tree(selected_tree, feature_names=feature_names, class_names=[str(i) for i in np.unique(y)],
          filled=True, rounded=True)
plt.title(f"Decision Tree {tree_index_to_plot + 1} within Random Forest")
plt.show()