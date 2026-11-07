
import json
import joblib
import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss
)

from sklearn.calibration import calibration_curve


# Fix the random seed so results can be reproduced
RANDOM_STATE = 42


# Load the training and validation datasets
train_data = pd.read_csv("data/processed/train.csv")
validation_data = pd.read_csv("data/processed/validation.csv")


# Select only features available when the appointment is booked
# reminder_sms_planned is included only if the reminder is decided at booking
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


# Separate input features and target values
X_train = train_data[all_features]
y_train = train_data["no_show"]

X_validation = validation_data[all_features]
y_validation = validation_data["no_show"]


# Fill missing numerical values and scale them for Logistic Regression
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# Fill missing categories and convert text into numerical columns
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# Combine numerical and categorical preprocessing
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# Define a simple baseline and two machine learning models
models = {
    "Baseline": DummyClassifier(strategy="most_frequent"),

    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
}


# Store model results so we can compare their performance
results = {}


# Train and evaluate each model using the same preprocessing steps
for model_name, classifier in models.items():

    print("\nTraining:", model_name)

    # Keep preprocessing and model together in one pipeline
    model_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier)
    ])

    # Learn patterns using only the training dataset
    model_pipeline.fit(X_train, y_train)

    # Predict probabilities and classes on validation data
    predicted_probabilities = model_pipeline.predict_proba(
        X_validation
    )[:, 1]

    predicted_classes = (predicted_probabilities >= 0.5).astype(int)

    # Calculate metrics to compare model performance
    precision = precision_score(
        y_validation, predicted_classes, zero_division=0
    )

    recall = recall_score(
        y_validation, predicted_classes, zero_division=0
    )

    f1 = f1_score(
        y_validation, predicted_classes, zero_division=0
    )

    matrix = confusion_matrix(
        y_validation, predicted_classes, labels=[0, 1]
    )

    # Calculate Brier score to check probability calibration
    brier = brier_score_loss(
        y_validation, predicted_probabilities
    )

    # Calculate calibration values for the probability predictions
    actual_rates, predicted_rates = calibration_curve(
        y_validation,
        predicted_probabilities,
        n_bins=5,
        strategy="uniform"
    )

    # Save the evaluation results for this model
    results[model_name] = {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": matrix.tolist(),
        "brier_score": float(brier),
        "calibration_actual_rates": actual_rates.tolist(),
        "calibration_predicted_rates": predicted_rates.tolist()
    }

    # Display the results in the terminal
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("Confusion Matrix:\n", matrix)
    print("Brier Score:", round(brier, 4))

    print("Calibration - actual rates:", actual_rates)
    print("Calibration - predicted rates:", predicted_rates)


# Select the best model using validation F1 score
best_model_name = max(
    ["Logistic Regression", "Random Forest"],
    key=lambda name: results[name]["f1_score"]
)


# Retrain the selected model using training and validation data
X_combined = pd.concat([X_train, X_validation])
y_combined = pd.concat([y_train, y_validation])

best_classifier = models[best_model_name]

final_model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", best_classifier)
])

final_model.fit(X_combined, y_combined)


# Create the models folder before saving the model
Path("models").mkdir(parents=True, exist_ok=True)


# Save preprocessing and classifier together in one file
joblib.dump(final_model, "models/model.joblib")


# Save comparison results so they can be used in the report
with open("models/metrics.json", "w") as file:
    json.dump(results, file, indent=4)


# Display the selected model and saved files
print("\n" + "=" * 50)
print("MODEL TRAINING COMPLETED")
print("=" * 50)

print("Best model:", best_model_name)
print("Saved model: models/model.joblib")
print("Saved metrics: models/metrics.json")
