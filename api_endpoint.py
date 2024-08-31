from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved pipeline
model = joblib.load('model_pipeline.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    json_data = request.get_json()
    input_data = pd.DataFrame(json_data)
    predictions = model.predict(input_data)
    prediction_probs = model.predict_proba(input_data)[:, 1]
    return jsonify({
        'predictions': predictions.tolist(),
        'probabilities': prediction_probs.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
