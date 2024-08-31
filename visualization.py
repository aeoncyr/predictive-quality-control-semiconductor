import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Load the saved pipeline
loaded_pipeline = joblib.load('model_pipeline.pkl')

# Load data
data_file_path = 'dataset/secom.data'
labels_file_path = 'dataset/secom_labels.data'

data = pd.read_csv(data_file_path, sep='\s+', header=None)
labels = pd.read_csv(labels_file_path, sep='\s+', header=None, usecols=[0])
labels = labels[0].apply(lambda x: 1 if x == -1 else 0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42, stratify=labels)

# Predict on the test set
y_pred = loaded_pipeline.predict(X_test)

# Evaluation
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(report)

print("\nAccuracy Score:")
print(accuracy)

# Visualization
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Predict on new data
new_data = X_test[:10]
predictions = loaded_pipeline.predict(new_data)
prediction_probs = loaded_pipeline.predict_proba(new_data)[:, 1]

# Create a DataFrame to save the results
results_df = pd.DataFrame(new_data)
results_df['Predicted_Label'] = predictions
results_df['Probability_Class_1'] = prediction_probs

# Save predictions to a CSV file
results_df.to_csv('predictions.csv', index=False)

print("\nPredictions on New Data:")
print(results_df)