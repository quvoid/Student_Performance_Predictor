from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
try:
    model_path = os.path.join(os.path.dirname(__file__), '../models/student_performance_model.pkl')
    model = joblib.load(model_path)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model not found at {model_path}")
    model = None

# Mappings (Must match training/inference logic)
GENDER_MAP = {'Female': 0, 'Male': 1}
PARENTAL_MAP = {'High': 0, 'Low': 1, 'Medium': 2}
ONLINE_MAP = {False: 0, True: 1, 'False': 0, 'True': 1}

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        
        # Extract features from JSON
        gender = GENDER_MAP.get(data.get('Gender'))
        attendance = float(data.get('AttendanceRate'))
        study_hours = float(data.get('StudyHoursPerWeek'))
        prev_grade = float(data.get('PreviousGrade'))
        extra_activities = int(data.get('ExtracurricularActivities'))
        parental = PARENTAL_MAP.get(data.get('ParentalSupport'))
        online = ONLINE_MAP.get(data.get('Online Classes Taken'))

        # Validate Inputs
        if None in [gender, parental, online]:
             return jsonify({'error': 'Invalid categorical value provided.'}), 400

        # Create feature vector
        features = np.array([[gender, attendance, study_hours, prev_grade, extra_activities, parental, online]])

        # Predict
        prediction_class = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        result = "High Performance" if prediction_class == 1 else "Average/Low Performance"

        return jsonify({
            'prediction': result,
            'probability': float(probability),
            'student_data': data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
