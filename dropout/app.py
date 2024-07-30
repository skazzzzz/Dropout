from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load your trained model, encoder classes, and scaler
model = joblib.load('model.pkl')
encoder_classes = joblib.load('encoder_classes.pkl')
scaler = joblib.load('scaler.pkl')

# List of columns used for the prediction
columns = [
    'Age at enrollment', 'Marital status', 'Course', 'Daytime/evening attendance',
    'Previous qualification', "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation", 'Displaced', 'Educational special needs',
    'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International'
]

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()
    input_data = {col: [form_data.get(col, 'unknown')] for col in columns}
    input_df = pd.DataFrame(input_data)

    # Preprocess the input data
    categorical_features = [
        'Marital status', 'Course', 'Daytime/evening attendance',
        'Previous qualification', "Mother's qualification", "Father's qualification",
        "Mother's occupation", "Father's occupation", 'Displaced', 'Educational special needs',
        'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International'
    ]

    for feature in categorical_features:
        classes = encoder_classes[feature]
        encoder = LabelEncoder()
        encoder.classes_ = np.append(classes, 'unknown')
        input_df[feature] = input_df[feature].apply(lambda x: x if x in classes else 'unknown')
        input_df[feature] = encoder.transform(input_df[feature])

    numerical_features = ['Age at enrollment']
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])

    # Predict using the model
    prediction = model.predict(input_df)
    result = 'Dropout' if prediction[0] == 0 else 'Enrolled' if prediction[0] == 1 else 'Graduate'
    return f'The predicted outcome is: {result}'

if __name__ == '__main__':
    app.run(debug=True)
