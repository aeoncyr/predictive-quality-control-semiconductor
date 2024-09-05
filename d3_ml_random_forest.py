import pickle
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, make_scorer, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load data
with open('dataset/secom_data_smoted.pkl', 'rb') as file:
    secom_data = pickle.load(file)
with open('dataset/secom_pca_optimal.pkl', 'rb') as file:
    secom_data_pca = pickle.load(file)
with open('dataset/secom_labels_smoted.pkl', 'rb') as file:
    secom_labels = pickle.load(file).squeeze()

print(secom_data.head())
print(secom_data_pca.head())
print(secom_labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(secom_data, secom_labels, test_size=0.3, random_state=42)

# Train-test split PCA
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(secom_data_pca, secom_labels, test_size=0.3, random_state=42)

## Model definition

# Define Base RF
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')

# Wrap the SVC model with Bagging
bagging_rf = BaggingClassifier(estimator=rf_model, random_state=42)

# Use StratifiedKFold to maintain the proportion of classes in each fold
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Change evaluation metrics to F1-score
f1_scorer = make_scorer(f1_score, pos_label=0)

# Define grid parameters
param_grid = {
    'estimator__n_estimators': [100, 200, 500],              # Number of trees in the forest
    'estimator__max_features': ['sqrt', 'log2'],     # Number of features to consider at each split
    'estimator__max_depth': [None, 10, 20, 30],              # Maximum depth of the tree
    'estimator__min_samples_split': [2, 5, 10],              # Minimum number of samples required to split an internal node
    'estimator__min_samples_leaf': [1, 2, 4],                # Minimum number of samples required to be at a leaf node
}

# Train using the smoted data first

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=bagging_rf, param_grid=param_grid, cv=stratified_kfold, scoring=f1_scorer, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters found
print(f"Best parameters found: {grid_search.best_params_}")

# Use the best found model
best_rf_model = grid_search.best_estimator_

# Predict and evaluate the model
rf_pred_best = best_rf_model.predict(X_test)
print("Best SVM Performance:")
print(f"Accuracy: {accuracy_score(y_test, rf_pred_best):.4f}")
print(f"Precision: {precision_score(y_test, rf_pred_best, pos_label=1):.4f}")
print(f"Recall: {recall_score(y_test, rf_pred_best, pos_label=1):.4f}")
print(f"F1-Score: {f1_score(y_test, rf_pred_best, pos_label=1):.4f}")

conf_matrix = confusion_matrix(y_test, rf_pred_best)
f1_score_smoted = f1_score(y_test, rf_pred_best)
report = classification_report(y_test, rf_pred_best)

print("Confusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(report)

print("\n F1 Score:")
print(f1_score_smoted)

# Now train using the PCA data

# Perform grid search with cross-validation
grid_search_pca = GridSearchCV(estimator=bagging_rf, param_grid=param_grid, cv=stratified_kfold, scoring=f1_scorer, verbose=2)
grid_search_pca.fit(X_train_pca, y_train_pca)

# Best parameters found
print(f"Best parameters found: {grid_search_pca.best_params_}")

# Use the best found model
best_rf_model_pca = grid_search_pca.best_estimator_

# Predict and evaluate the model
rf_pred_best_pca = best_rf_model_pca.predict(X_test_pca)
print("Best SVM Performance:")
print(f"Accuracy: {accuracy_score(y_test_pca, rf_pred_best_pca):.4f}")
print(f"Precision: {precision_score(y_test_pca, rf_pred_best_pca, pos_label=1):.4f}")
print(f"Recall: {recall_score(y_test_pca, rf_pred_best_pca, pos_label=1):.4f}")
print(f"F1-Score: {f1_score(y_test_pca, rf_pred_best_pca, pos_label=1):.4f}")

conf_matrix_pca = confusion_matrix(y_test_pca, rf_pred_best_pca)
f1_score_pca = f1_score(y_test_pca, rf_pred_best_pca)
report_pca = classification_report(y_test_pca, rf_pred_best_pca)

print("Confusion Matrix:")
print(conf_matrix_pca)

print("\nClassification Report:")
print(report_pca)

print("\n F1 Score:")
print(f1_score_pca)

# Save the models as pickle file

joblib.dump(best_rf_model, 'trained_models/rf_model.pkl')
joblib.dump(best_rf_model_pca, 'trained_models/rf_model_pca.pkl')

# Also save best grid parameters as JSON for references
with open('trained_models/params/rf_model_params.json', 'w') as file:
    json.dump(grid_search.best_params_, file)

with open('trained_models/params/rf_model_pca_params.json', 'w') as file:
    json.dump(grid_search_pca.best_params_, file)