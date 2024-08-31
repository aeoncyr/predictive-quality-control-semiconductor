import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib

# Load data
data_file_path = 'dataset/secom.data'
labels_file_path = 'dataset/secom_labels.data'

data = pd.read_csv(data_file_path, sep='\s+', header=None)
labels = pd.read_csv(labels_file_path, sep='\s+', header=None, usecols=[0])
labels = labels[0].apply(lambda x: 1 if x == -1 else 0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42, stratify=labels)

# Build the Pipeline
pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

#Train the Model
param_grid = {
    'classifier__n_estimators': [100, 200, 500],
    'classifier__max_depth': [None, 5, 8, 10, 20, 50],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__max_features': ['sqrt', 'log2'],
    'classifier__criterion' :['gini', 'entropy', 'log_loss']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Show best param
print(grid_search.best_params_)

# Best model
best_model = grid_search.best_estimator_

# Save the Pipeline
joblib.dump(best_model, 'model_pipeline.pkl')