import pandas as pd
from flask import Flask, render_template, request
import pickle
import warnings
warnings.filterwarnings(action='ignore')

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("lr.pkl", "rb"))

# Selected features and their expected types
features = {
    'Lot Area (in Sqft)': float,
    'No of Floors':int,
    'No of Bathrooms':int,
    'No of Bedrooms':int,
    'Overall Grade':float,
    'Area of the House from Basement (in Sqft)':float,
    'Basement Area (in Sqft)':float,
    'Living Area after Renovation (in Sqft)':float,
    'Age of House (in Years)':int,
    'Latitude':float,
    'Longitude':float
}

@app.route('/')
def home():
    return render_template('house_pred_index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convert input form values to appropriate types
        input_data = [features[key](request.form.get(key)) for key in features]
        # Make prediction
        # lr.predict(input)
        prediction = model.predict([input_data])[0]
        print(prediction)
        return render_template('result.html', prediction=round(prediction, 2))


    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
