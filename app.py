from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# load the model
model = joblib.load('model.pkl')  # Ensure this path is correct

# initialize the flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # get data from POST request
    data = request.json
    
    # check that we have the right input
    if 'features' not in data:
        return jsonify({'error': 'No features provided'}), 400

    # convert input into a pandas df
    features = pd.DataFrame(data['features']).T  # transpose to make it a single sample DataFrame
    features.columns = [
        'LIMIT_BAL', 'AGE', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
        'SEX_Female', 'SEX_Male', 'EDU_Graduate_School', 'EDU_High_School',
        'EDU_Others', 'EDU_University', 'MARRIAGE_Married', 'MARRIAGE_Others',
        'MARRIAGE_Single'
    ]

    # predict
    prediction = model.predict(features)

    # convert the prediction to json
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
