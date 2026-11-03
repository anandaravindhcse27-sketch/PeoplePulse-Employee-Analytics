# Employee Performance Analytics System

A beginner-friendly FDS project using Pandas, feature engineering, and Scikit-learn classification.

## Features
- Sample employee dataset included
- CSV upload support
- Pandas data cleaning and feature engineering
- Interactive Streamlit dashboard and charts
- Random Forest classification with test accuracy, classification report, and confusion matrix
- Employee performance prediction form
- Download cleaned data

## Run in VS Code
1. Install Python 3.10 or newer.
2. Open this folder in VS Code.
3. Open Terminal → New Terminal.
4. Create a virtual environment (Windows):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate
   ```
5. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
6. Run:
   ```powershell
   python -m streamlit run app.py
   ```
7. Streamlit opens the app in your browser, usually at http://localhost:8501.

## CSV columns
Department, Years_Experience, Training_Hours, Attendance_Rate, Projects_Completed, Manager_Rating, Monthly_Salary.
An optional `Performance_Label` column can contain labels such as Low, Medium, High. If absent, demo labels are generated from a simple score.

## Note
This is an educational demonstration. Do not use predictions as the sole basis for real employment decisions.
