import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Address the data
data_path = 'dataset/secom.data'
labels_path = 'dataset/secom_labels.data'

# Load the SECOM data and label
secom_data = pd.read_csv(data_path, sep='\s+', header=None)
secom_labels = pd.read_csv(labels_path, sep='\s+', header=None, usecols=[0], names=['Label'])

# Display first 5 data and labels in the dataset
print("First 5 data: \n")
print(secom_data.head())

print("\n First 5 labels: \n")
print(secom_labels.head())

# Display class distribution
target_distribution = secom_labels.iloc[:, 0].value_counts()

sns.barplot(x=target_distribution.index, y=target_distribution.values)
plt.title("Distribution of Pass/Fail Labels")
plt.ylabel("Proportion")
plt.xlabel("Quality Control Result")
plt.xticks(ticks=[0, 1], labels=["Pass (-1)", "Fail (1)"])
plt.show()

# Display missing value distribution
missing_values_percentage = pd.DataFrame(secom_data.isnull().mean() * 100, columns=['MissingPercentage'])
missing_values_percentage[missing_values_percentage['MissingPercentage'] > 0].hist(bins=30, color='orange')
plt.title("Distribution of Missing Values Across Features")
plt.xlabel("Percentage of Missing Values")
plt.ylabel("Number of Features")
plt.show()

# Display statistic summary
summary_statistics = secom_data.describe().T
print(summary_statistics.head())