import joblib
import numpy as np
import pandas as pd
import os

def predict_student_performance(student_data):
    """
    Predicts student performance based on input dictionary.
    
    Args:
        student_data (dict): Dictionary containing student features:
                             'Gender', 'AttendanceRate', 'StudyHoursPerWeek', 
                             'PreviousGrade', 'ExtracurricularActivities', 
                             'ParentalSupport', 'Online Classes Taken'
    
    Returns:
        str: Prediction result string.
    """
    
    # Load Model
    # Note: In a production app, you'd load this once at startup
    try:
        model_path = os.path.join(os.path.dirname(__file__), "../models/student_performance_model.pkl")
        model = joblib.load(model_path)
    except FileNotFoundError:
        return f"Error: Model file '{model_path}' not found. Please train the model first."

    # Manual Encoding Mappings (Must match training LabelEncoder)
    gender_map = {'Female': 0, 'Male': 1}
    # Alphabetical order: High, Low, Medium -> 0, 1, 2
    parental_map = {'High': 0, 'Low': 1, 'Medium': 2} 
    online_map = {False: 0, True: 1, 'False': 0, 'True': 1}

    # Extract and Encode Features
    try:
        gender = gender_map[student_data['Gender']]
        attendance = float(student_data['AttendanceRate'])
        study_hours = float(student_data['StudyHoursPerWeek'])
        prev_grade = float(student_data['PreviousGrade'])
        extra_activities = int(student_data['ExtracurricularActivities'])
        parental = parental_map[student_data['ParentalSupport']]
        online = online_map[student_data['Online Classes Taken']]
        
        # Create Feature Vector
        # Order must match columns in X_train:
        # ['Gender', 'AttendanceRate', 'StudyHoursPerWeek', 'PreviousGrade',
        #  'ExtracurricularActivities', 'ParentalSupport', 'Online Classes Taken']
        features = np.array([[gender, attendance, study_hours, prev_grade, extra_activities, parental, online]])
        
        # Predict
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1] # Prob of class 1
        
        # Interpret Result
        if prediction == 1:
            return f"High Performance (Probability: {probability:.2f})"
        else:
            return f"Average/Low Performance (Probability: {probability:.2f})"

    except KeyError as e:
        return f"Error: Missing or invalid value for field {e}"
    except Exception as e:
        return f"Error during prediction: {e}"

if __name__ == "__main__":
    # Test Case 1: Likely High Performer
    student1 = {
        'Gender': 'Female',
        'AttendanceRate': 95,
        'StudyHoursPerWeek': 25,
        'PreviousGrade': 90,
        'ExtracurricularActivities': 2,
        'ParentalSupport': 'High',
        'Online Classes Taken': True
    }
    
    # Test Case 2: Likely Low Performer
    student2 = {
        'Gender': 'Male',
        'AttendanceRate': 60,
        'StudyHoursPerWeek': 5,
        'PreviousGrade': 50,
        'ExtracurricularActivities': 0,
        'ParentalSupport': 'Low',
        'Online Classes Taken': False
    }

    print("--- Prediction Test ---")
    print(f"Student 1: {predict_student_performance(student1)}")
    print(f"Student 2: {predict_student_performance(student2)}")
