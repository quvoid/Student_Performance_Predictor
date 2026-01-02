# 🎓 Student Performance Prediction Web App

A full-stack Machine Learning web application designed to predict student performance based on academic and behavioral metrics. This project demonstrates the end-to-end integration of a Machine Learning pipeline with a responsive web interface.

## 🚀 Project Overview

This application takes various student parameters (such as study hours, attendance, and parental support) and predicts whether a student is likely to have **High Performance** or **Average/Low Performance**. It serves as a tool for educators to identify students who might need additional support.

## 📊 Dataset

The model is trained on the **Student Performance Prediction Dataset** sourced from Kaggle.
*   **Source**: [Kaggle - Student Performance Predictions](https://www.kaggle.com/datasets/haseebindata/student-performance-predictions)
*   **Features**: Attendance Rate, Study Hours, Previous Grades, Extracurricular Activities, Parental Support, etc.
*   **Target**: Final Grade (Classified into High vs. Low/Average).

## 🧠 Machine Learning Concepts

*   **Data Preprocessing**: Handling missing values (Median/Mode imputation), Label Encoding for categorical variables.
*   **Model Selection**: Logistic Regression (chosen for interpretability and baseline performance).
*   **Evaluation**: Accuracy, Precision, Recall, and F1-Score analysis.
*   **Inference**: Real-time prediction probabilities served via API.

## 🛠️ Tech Stack

*   **Frontend**: HTML5, JavaScript (Fetch API), **Tailwind CSS** (for responsive UI).
*   **Backend**: **Flask** (Python) for the REST API.
*   **Machine Learning**: **Scikit-learn**, Pandas, NumPy, Joblib.
*   **Version Control**: Git & GitHub.

## 💻 How to Run Locally

Follow these steps to set up the project on your local machine.

### 1. Clone the Repository
```bash
git clone https://github.com/quvoid/Student_Performance_Predictor.git
cd Student_Performance_Predictor
```

### 2. Install Dependencies
Ensure you have Python installed. It's recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

### 3. Run the Application
Navigate to the backend directory and start the Flask server:
```bash
cd backend
python app.py
```

### 4. Access the Web App
Open your browser and verify the app is running at:
**[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

## 📸 Screenshots

*(Add screenshots of the web interface here)*

## 🔮 Future Improvements

*   **Advanced Models**: Experiment with Random Forest or XGBoost for better accuracy.
*   **Data Visualization**: Add charts to visualize student data distribution on the dashboard.
*   **User Authentication**: Allow teachers to save and track student records over time.
*   **Deployment**: Deploy the application to a cloud platform like Render or Vercel.

---
*Built with ❤️ by [Your Name]*
