
# Hospital Appointment No-Show Prediction

## 1. Project Overview

The Hospital Appointment No-Show Prediction project is a machine learning application that estimates the probability of a patient missing a scheduled hospital appointment.

The system uses appointment-related information available at booking time to generate a prediction. The results can help hospital staff identify appointments that may need additional attention.

**This project is designed for human-reviewed scheduling support only. It does not make medical decisions.**

## 2. Project Objectives

- Predict the probability of an appointment no-show.
- Generate a synthetic appointment dataset for model development.
- Compare multiple machine learning models.
- Evaluate model performance using appropriate classification metrics.
- Deploy the trained model through a Flask web application.
- Maintain a reproducible preprocessing and training pipeline.

## 3. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Flask
- Joblib
- HTML and CSS
- Git and GitHub

## 4. Dataset

The project uses a synthetically generated hospital appointment dataset.

- Original generated records: 6,015
- Records after duplicate removal: 6,000
- Features: 22 columns
- Target variable: `no_show`

Target values:
- `0` – Appointment attended
- `1` – Appointment missed

The dataset includes appointment timing, department, booking channel, reminder information, and previous appointment history.

No real patient records or diagnosis information are used.

## 5. Data Preprocessing

The preprocessing process includes:

- Removing duplicate appointment IDs.
- Handling missing and invalid records.
- Creating a previous-attended-visit indicator.
- Sorting records by booking datetime.
- Splitting the data chronologically.

Dataset split:

| Dataset | Records |
|---|---:|
| Training | 4,200 |
| Validation | 900 |
| Testing | 900 |
| Total | 6,000 |

## 6. Machine Learning Models

Three approaches are compared:

1. Dummy Classifier – Baseline model.
2. Logistic Regression – Classification model with balanced class weights.
3. Random Forest – Ensemble classification model with balanced class weights.

A Scikit-learn Pipeline and ColumnTransformer are used to combine preprocessing and model training.

The selected model is saved as `models/model.joblib`.

## 7. Model Evaluation

The final test-set results for the selected Logistic Regression model are:

| Metric | Result |
|---|---:|
| Precision | 0.2746 |
| Recall | 0.6376 |
| F1 Score | 0.3838 |
| Brier Score | 0.2255 |

Confusion matrix:

```text
[[500 251]
 [ 54  95]]
```

The model detected 95 out of 149 no-show appointments in the test dataset.

The results also show that there are false positives and missed no-shows. Therefore, predictions should be reviewed by a human and should not be treated as certain outcomes.

A calibration curve is saved at `reports/figures/calibration_curve.png`.

## 8. Flask Web Application

The application uses the saved machine learning pipeline to generate predictions.

Main workflow:

1. Open the Flask application.
2. Enter appointment information.
3. Submit the appointment form.
4. The application loads the saved model and calculates the no-show probability.
5. The result is displayed on the results page.

The application does not use hard-coded predictions.

## 9. Project Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/Aishakhan27/hospital-no-show-prediction.git
```

### Step 2: Open the project folder

```bash
cd hospital-no-show-prediction
```

### Step 3: Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 4: Run the Flask application

```bash
python -m app.app
```

Open this address in your browser:

```text
http://127.0.0.1:5000/
```

Use Python 3.13 for this project if multiple Python versions are installed.

## 10. Running the Tests

Run all automated tests using:

```bash
python -m unittest discover -s tests -v
```

The tests cover:
- Flask home page and prediction route.
- Saved model loading and prediction probability.
- Processed dataset files.
- Duplicate appointment IDs.
- Chronological data splitting.
- Valid target values.

## 11. Project Structure

```text
no-show-prediction/
│
├── app/
│   ├── app.py
│   ├── templates/
│   └── static/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── data_dictionary.md
│
├── models/
│   ├── model.joblib
│   └── metrics.json
│
├── notebooks/
├── reports/
├── src/
├── tests/
│
├── .gitignore
├── Makefile
├── README.md
└── requirements.txt
```

## 12. Limitations and Responsible Use

- The dataset is synthetic and does not represent actual hospital populations.
- Model performance on synthetic data does not establish performance in a real hospital.
- Predictions may contain false positives and false negatives.
- The model should be evaluated on appropriate real-world data before any operational use.
- Human review is required before taking any scheduling action.
- Appointment predictions must not be used to deny or restrict healthcare access.

## 13. Future Improvements

- Evaluate the model on a suitable de-identified real-world dataset.
- Improve probability calibration.
- Study model performance across relevant demographic groups.
- Improve the Flask application's validation and user interface.
- Explore different prediction thresholds based on operational requirements.

## 14. Author

Developed as a machine learning project for hospital appointment no-show prediction.

**GitHub Repository:**  
https://github.com/Aishakhan27/hospital-no-show-prediction
