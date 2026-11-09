
import unittest

# Import the Flask application so we can test its routes.
from app.app import app


class TestHospitalApp(unittest.TestCase):

    # Create a test client to send requests without opening a browser.
    def setUp(self):
        self.client = app.test_client()

    # Check whether the home page opens successfully.
    def test_home_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    # Check whether the prediction route accepts appointment details.
    def test_prediction(self):
        appointment_data = {
            "lead_time_days": "15",
            "appointment_hour": "10",
            "appointment_month": "6",
            "reminder_sms_planned": "1",
            "is_holiday_week": "0",
            "weather_risk_month": "0",
            "age": "30",
            "distance_km": "5",
            "prior_appointments": "5",
            "prior_no_shows": "1",
            "days_since_last_visit": "90",
            "has_previous_attended_visit": "1",
            "appointment_weekday": "Monday",
            "department": "Dermatology",
            "appointment_type": "Follow-up",
            "booking_channel": "Mobile App",
            "gender": "F",
            "payment_type": "Insurance"
        }

        # Send the appointment details to the prediction route.
        response = self.client.post("/predict", data=appointment_data)

        # Confirm that the prediction result page opens.
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Prediction Result", response.data)


if __name__ == "__main__":
    unittest.main()
