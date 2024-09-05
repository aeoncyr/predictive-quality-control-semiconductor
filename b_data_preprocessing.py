import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle

# Add data address
data_file_path = '.dataset/secom.data'
labels_file_path = 'dataset/secom_labels.data'

# Load the dataset
data_raw = pd.read_csv(data_file_path, sep='\s+', header=None)

# Open label
labels = pd.read_csv(labels_file_path, sep='\s+', header=None, usecols=[0], names=['Label'])
print("Labels before:\n")
print(labels)

# Modify label for easier interpretation
labels = labels.replace(1, 0)  # Change 1 to 0, fail = 0
labels = labels.replace(-1, 1)  # Change -1 to 1, pass = 1
print("\n Labels after: \n")
print(labels)

# Combine features and labels into a single DataFrame
data = pd.concat([data_raw, labels], axis=1)
print(data.head())

# Separate data by class
data_class_0 = data[data['Label'] == 0].drop('Label', axis=1)
data_class_1 = data[data['Label'] == 1].drop('Label', axis=1)

print("Class 0 data: \n")
print(data_class_0.head())

print("\n Class 1 data: \n")
print(data_class_1.head())

# Impute missing values for each class separately
imputer = SimpleImputer(strategy='mean')

# Impute class 0
data_class_0_imputed = imputer.fit_transform(data_class_0)
data_class_0_imputed = pd.DataFrame(data_class_0_imputed, columns=data_class_0.columns)
data_class_0_imputed['Label'] = 0  # Add the label column back

# Impute class 1
data_class_1_imputed = imputer.fit_transform(data_class_1)
data_class_1_imputed = pd.DataFrame(data_class_1_imputed, columns=data_class_1.columns)
data_class_1_imputed['Label'] = 1

# Combine the imputed data back together
data_imputed = pd.concat([data_class_0_imputed, data_class_1_imputed], axis=0)

print("Combined data before shuffle: \n")
print(data_imputed)

# Shuffle the data then reset the index
data_imputed = shuffle(data_imputed)
data_imputed = data_imputed.reset_index(drop=True)

print("\n \n Combined data after shuffle: \n")
print(data_imputed)

# Separate data and labels
labels_imputed = data_imputed['Label']
data_imputed = data_imputed.drop('Label', axis=1)

# Handle class imbalance with oversampling using SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(random_state=42)
data_smoted, labels_smoted = smote.fit_resample(data_imputed, labels_imputed)

# Display SMOTE result
target_distribution = labels_smoted.value_counts()
sns.barplot(x=target_distribution.index, y=target_distribution.values)
plt.title("Distribution of Pass/Fail Labels")
plt.ylabel("Proportion")
plt.xlabel("Quality Control Result")
plt.xticks(ticks=[0, 1], labels=["Pass (-1)", "Fail (1)"])
plt.show()

# Save to csv file
with open('dataset/secom_data_smoted.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(data_smoted), file)

with open('dataset/secom_labels_smoted.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(labels_smoted), file)