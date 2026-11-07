

import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    brier_score_loss
)

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt


# Load the saved model so we evaluate the actual trained pipeline
model = joblib.load("models/model.joblib")


# Load the untouched test dataset
test_data = pd.read_csv("data/processed/test.csv")


# Select only features used during training
numeric_features = [
    "lead_time_days",
    "appointment_hour",
    "appointment_month",
    "reminder_sms_planned",
    "is_holiday_week",
    "weather_risk_month",
    "age",
    "distance_km",
    "prior_appointments",
    "prior_no_shows",
    "days_since_last_visit",
    "has_previous_attended_visit"
]

categorical_features = [
    "appointment_weekday",
    "department",
    "appointment_type",
    "booking_channel",
    "gender",
    "payment_type"
]

all_features = numeric_features + categorical_features


# Separate test inputs and actual answers
X_test = test_data[all_features]
y_test = test_data["no_show"]


# Predict probabilities using the saved model
probabilities = model.predict_proba(X_test)[:, 1]


# Apply the default classification threshold
predictions = (probabilities >= 0.5).astype(int)


# Calculate evaluation metrics on unseen test records
precision = precision_score(
    y_test, predictions, zero_division=0
)

recall = recall_score(
    y_test, predictions, zero_division=0
)

f1 = f1_score(
    y_test, predictions, zero_division=0
)

matrix = confusion_matrix(
    y_test, predictions, labels=[0, 1]
)

brier = brier_score_loss(y_test, probabilities)


# Display all important evaluation results
print("=" * 50)
print("FINAL TEST SET EVALUATION")
print("=" * 50)

print("Test records:", len(test_data))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("Brier Score:", round(brier, 4))

print("\nConfusion Matrix:")
print(matrix)

print("\nClassification Report:")
print(classification_report(
    y_test, predictions, zero_division=0
))


# Create a calibration curve to compare predicted and actual probabilities
actual_rates, predicted_rates = calibration_curve(
    y_test,
    probabilities,
    n_bins=5,
    strategy="uniform"
)


# Plot the calibration curve to inspect probability reliability
plt.figure(figsize=(7, 5))

plt.plot(
    predicted_rates,
    actual_rates,
    marker="o",
    label="Logistic Regression"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect Calibration"
)

plt.xlabel("Mean Predicted Probability")
plt.ylabel("Actual No-Show Rate")
plt.title("Model Calibration Curve")
plt.legend()
plt.grid()

plt.savefig(
    "reports/figures/calibration_curve.png",
    bbox_inches="tight"
)

plt.show()


print("\nCalibration curve saved successfully!")
