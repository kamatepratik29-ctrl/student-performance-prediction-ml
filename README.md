# Student Performance Prediction & Analysis System

A beginner-friendly final AI & Machine Learning project using Python, Pandas, Scikit-learn, Matplotlib, and Streamlit.

## Important note
The included dataset is synthetic and generated for learning. Predictions are demonstrations only. Do not use them for real student evaluation, grading, admissions, or other consequential decisions.

## Features
- Explore and preview sample student records.
- Clean duplicate rows and missing/invalid values.
- Visualize score distribution, study-hours relationship, and feature correlations.
- Train a Random Forest Regressor.
- Evaluate using Mean Absolute Error (MAE) and R².
- Enter sample inputs and estimate a final score.
- Compare actual and predicted test-set scores.

## Tech stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Streamlit.

## Project structure
```text
student-performance-prediction/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    └── student_performance.csv
```

## Run on Windows PowerShell
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
Open the local URL shown by Streamlit, usually http://localhost:8501.

## Machine learning workflow
1. Load the CSV using Pandas.
2. Remove duplicates and clean invalid/missing records.
3. Select five input features and the final exam score target.
4. Split into training and testing data.
5. Train a Random Forest Regressor.
6. Evaluate with MAE and R².
7. Predict for new sample inputs.

## Dataset columns
- `study_hours_per_day`
- `attendance_percentage`
- `previous_exam_score`
- `assignment_average`
- `sleep_hours_per_day`
- `final_exam_score`

The 500 records are generated using a fixed random seed. They are illustrative, not real student data.

## Learning outcomes
Practised data loading, cleaning, visualization, regression, train/test splitting, model evaluation, and building an interactive ML app.

## Future improvements
Use a properly licensed real dataset, compare multiple algorithms, add cross-validation, and examine feature importance.
