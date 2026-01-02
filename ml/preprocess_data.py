import kagglehub
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def load_and_preprocess():
    print("Loading dataset...")
    # Download/Locate the dataset
    path = kagglehub.dataset_download("haseebindata/student-performance-predictions")
    
    # Find the CSV file
    csv_file = None
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith("student_performance_updated_1000.csv"):
                csv_file = os.path.join(root, file)
                break
    
    if csv_file is None:
        print("Error: Target CSV file not found.")
        return

    print(f"Reading file: {csv_file}")
    df = pd.read_csv(csv_file)

    # 1. Drop irrelevant and redundant columns
    # Name and StudentID are not useful for prediction
    # 'Study Hours' and 'Attendance (%)' were found to have anomalies/errors, 
    # so we use 'StudyHoursPerWeek' and 'AttendanceRate' instead.
    cols_to_drop = ['Name', 'StudentID', 'Study Hours', 'Attendance (%)']
    print(f"Dropping columns: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # 2. Handle Missing Values
    print("Handling missing values...")
    
    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Impute numeric with median
    # We use median because it's more robust to outliers
    imputer_num = SimpleImputer(strategy='median')
    df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])
    
    # Impute categorical with most frequent value (mode)
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

    # 3. Encode Categorical Variables
    print("Encoding categorical variables...")
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"Encoded '{col}': {list(le.classes_)}")

    # 4. Separate Features and Target
    target_col = 'FinalGrade'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 5. Train-Test Split (80% Train, 20% Test)
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nProcessing Complete!")
    print(f"Training Data Shape: {X_train.shape}")
    print(f"Testing Data Shape: {X_test.shape}")
    
    # Optional: Preview the processed data
    print("\nSample of Processed Training Data:")
    print(X_train.head())

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    load_and_preprocess()
