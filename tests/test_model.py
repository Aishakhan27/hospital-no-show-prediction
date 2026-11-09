
import unittest
import pandas as pd
import joblib
from pathlib import Path

# Find the project folder and load the saved model.
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
model = joblib.load(MODEL_PATH)


class TestHospitalModel(unittest.TestCase):

    # Check whether the saved model exists.
    def test_model_file_exists(self):
        self.assertTrue(MODEL_PATH.exists())

    # Check whether the model returns a valid probability.
    def test_model_prediction(self):
        appointment = pd.DataFrame([{
            "lead_time_days": 15,
            "appointment_hour": 10,
            "appointment_month": 6,
            "reminder_sms_planned": 1,
            "is_holiday_week": 0,
            "weather_risk_month": 0,
            "age": 30,
            "distance_km": 5,
            "prior_appointments": 5,
            "prior_no_shows": 1,
            "days_since_last_visit": 90,
            "has_previous_attended_visit": 1,
            "appointment_weekday": "Monday",
            "department": "Dermatology",
            "appointment_type": "Follow-up",
            "booking_channel": "Mobile App",
            "gender": "F",
            "payment_type": "Insurance"
        }])

        # Get the probability of an appointment being missed.
        probability = model.predict_proba(appointment)[0][1]

        # Confirm that the probability is between zero and one.
        self.assertGreaterEqual(probability, 0)
        self.assertLessEqual(probability, 1)


if __name__ == "__main__":
    unittest.main()

