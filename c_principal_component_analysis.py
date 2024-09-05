import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the preprocessed data
with open("dataset/secom_data_smoted.pkl", "rb") as file:
    secom_data = pickle.load(file)
print(secom_data)

# Normalize/scale the data
scaler = StandardScaler().set_output(transform="pandas")
secom_data_scaled = scaler.fit_transform(secom_data)
print(secom_data_scaled)

# Apply PCA
pca = PCA().set_output(transform="pandas")
secom_pca = pca.fit_transform(secom_data_scaled)
print(secom_pca)

# Calculate the explained variance ratio for each principal component
explained_variance_ratio = pca.explained_variance_ratio_
print(explained_variance_ratio)

# Calculate the cumulative explained variance
cumulative_variance = np.cumsum(explained_variance_ratio)
print(cumulative_variance)

# Plot the explained variance and cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', color='b')
plt.title('Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()

# Determine the number of components needed to explain at least 95% of the variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Number of components to retain 95% variance: {n_components_95}")

# Apply PCA again with the optimal number of components
pca_optimal = PCA(n_components=n_components_95).set_output(transform="pandas")
secom_pca_optimal = pca_optimal.fit_transform(secom_data_scaled)
print(secom_pca_optimal)

# Save the PCA transformed data to a file using pickle
with open("dataset/secom_pca_optimal.pkl", "wb") as file:
    pickle.dump(secom_pca_optimal, file)