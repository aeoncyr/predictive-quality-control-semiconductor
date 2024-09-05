import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load saved models
svm_model = joblib.load('trained_models/svm_model.pkl')
sgd_model = joblib.load('trained_models/sgd_model.pkl')
rf_model = joblib.load('trained_models/rf_model.pkl')
svm_model_pca = joblib.load('trained_models/svm_model_pca.pkl')
sgd_model_pca = joblib.load('trained_models/sgd_model_pca.pkl')
rf_model_pca = joblib.load('trained_models/rf_model_pca.pkl')

# Define the reference SECOM data structure for compatibility checks
secom_reference_data = pd.read_csv('dataset/secom.data', sep='\s+', header=None)

def deploy_model(model='RF', data=None, verbose=False, pca=True):
    """
    Deploy the chosen machine learning model for prediction.

    Parameters:
    -----------
    model_choice : str, optional (default='RF')
        The model to use for prediction. Choices are:
        - 'SVM': Support Vector Machine model
        - 'SGD': Stochastic Gradient Descent Logistic Regression model
        - 'RF' : Random Forest model

    data : pd.DataFrame or np.array
        The data that needs to be predicted. Can be a single row (stream) or multiple rows.
        The number of features must match the reference SECOM data.

    verbose : bool, optional (default=False)
        If True, print detailed information during the process.

    use_pca : bool, optional (default=True)
        If True, apply PCA transformation to the data before prediction.

    Returns:
    --------
    predictions : np.ndarray
        The predicted labels for the input data.
    """
    # Choose the model
    if pca:
        model_dict = {
            'SVM': svm_model_pca,
            'SGD': sgd_model_pca,
            'RF': rf_model_pca
        }
    else: 
        model_dict = {
            'SVM': svm_model,
            'SGD': sgd_model,
            'RF': rf_model
        }

    model = model_dict.get(model, rf_model_pca)  # Default to RF model
    
    # Convert single row of data to the correct format (ensure it's 2D)
    if isinstance(data, pd.DataFrame) or isinstance(data, np.ndarray):
        if data.ndim == 1:  # Single row, reshape it to (1, -1)
            data = data.reshape(1, -1) if isinstance(data, np.ndarray) else data.values.reshape(1, -1)

    # Check if data is compatible with reference SECOM data (feature length match)
    if data.shape[1] != secom_reference_data.shape[1]:
        raise ValueError(f"Data has {data.shape[1]} features, but expected {secom_reference_data.shape[1]} features.")
    
    if verbose:
        print(f"Chosen model: {model}")

    # Check for missing features and impute them if necessary
    missing_columns = set(secom_reference_data.columns) - set(range(data.shape[1]))
    if missing_columns:
        if verbose:
            print(f"Missing features: {missing_columns}")
        # Add missing columns with NaN values
        for col in missing_columns:
            data[col] = np.nan

    # Impute missing values using the same imputation strategy used during training
    imputer = SimpleImputer(strategy='mean').set_output(transform="pandas")
    data_imputed = imputer.fit_transform(data)

    # Normalize/scale the input data to improve the result
    scaler = StandardScaler().set_output(transform="pandas")
    data_scaled = scaler.fit_transform(data_imputed)

    # Apply PCA if specified
    if pca:
        pca = PCA(n_components=158).set_output(transform="pandas")  # same as training PCA component size
        data_transformed = pca.fit_transform(data_scaled)
        if verbose:
            print("PCA applied.")
    else:
        data_transformed = data_scaled
        if verbose:
            print("PCA not applied.")

    # Predict using the chosen model
    predictions = model.predict(data_transformed)
    
    # Display results
    if verbose:
        print("Predictions:")
        print(predictions)

    return predictions

# Let's try the function
predict_result = deploy_model(data=secom_reference_data, verbose=True)