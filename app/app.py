
from flask import Flask, render_template, request
import pandas as pd
import joblib
from pathlib import Path

# Find the main project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Load the trained model
model = joblib.load(BASE_DIR / "models" / "model.joblib")

# Create the Flask application
app = Flask(__name__)


# Display the appointment input form
@app.route("/")
def home():
    return render_template("index.html")


# Receive appointment details and predict no-show probability
@app.route("/predict", methods=["POST"])
def predict():

    # Read numeric values entered in the form
    numeric_data = {
        "lead_time_days": float(request.form["lead_time_days"]),
        "appointment_hour": int(request.form["appointment_hour"]),
        "appointment_month": int(request.form["appointment_month"]),
        "reminder_sms_planned": int(request.form["reminder_sms_planned"]),
        "is_holiday_week": int(request.form["is_holiday_week"]),
        "weather_risk_month": int(request.form["weather_risk_month"]),
        "age": float(request.form["age"]),
        "distance_km": float(request.form["distance_km"]),
        "prior_appointments": int(request.form["prior_appointments"]),
        "prior_no_shows": int(request.form["prior_no_shows"]),
        "days_since_last_visit": (
            float(request.form["days_since_last_visit"])
            if request.form["days_since_last_visit"].strip()
            else float("nan")
        ),
        "has_previous_attended_visit": int(
            request.form["has_previous_attended_visit"]
        )
    }

    # Read categorical values entered in the form
    categorical_data = {
        "appointment_weekday": request.form["appointment_weekday"],
        "department": request.form["department"],
        "appointment_type": request.form["appointment_type"],
        "booking_channel": request.form["booking_channel"],
        "gender": request.form["gender"],
        "payment_type": request.form["payment_type"]
    }

    # Combine all input features into one row
    input_data = pd.DataFrame([{**numeric_data, **categorical_data}])

    # Get the model's predicted probability of a no-show
    probability = model.predict_proba(input_data)[0][1]

    # Convert the probability into a percentage
    probability_percent = round(probability * 100, 2)

    # Display the prediction result
    return render_template(
        "results.html",
        probability=probability_percent
    )


# Run the application locally
if __name__ == "__main__":
    app.run(debug=True)
