
import pandas as pd
import numpy as np


# Load the raw dataset so we can check its quality before cleaning
df = pd.read_csv("data/raw/hospital_appointments.csv")

print("=" * 55)
print("HOSPITAL APPOINTMENT DATASET VALIDATION")
print("=" * 55)


# Check the number of records and columns
print("\n1. DATASET SIZE")
print("Total records:", df.shape[0])
print("Total columns:", df.shape[1])


# Check whether all required columns are available
print("\n2. REQUIRED COLUMNS")

required_columns = [
    "appointment_id",
    "patient_id",
    "booking_datetime",
    "appointment_datetime",
    "lead_time_days",
    "age",
    "distance_km",
    "prior_appointments",
    "prior_no_shows",
    "days_since_last_visit",
    "no_show"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

print("Missing columns:", missing_columns)


# Check missing values because they need proper preprocessing
print("\n3. MISSING VALUES")
print(df.isnull().sum())


# Check duplicate rows and appointment IDs
print("\n4. DUPLICATE CHECKS")
print("Duplicate rows:", df.duplicated().sum())
print("Duplicate appointment IDs:",
      df["appointment_id"].duplicated().sum())


# Check data types to identify columns needing conversion
print("\n5. DATA TYPES")
print(df.dtypes)


# Check target distribution to understand class imbalance
print("\n6. TARGET DISTRIBUTION")
print(df["no_show"].value_counts())
print("\nTarget percentages:")
print(df["no_show"].value_counts(normalize=True) * 100)


# Convert dates so chronological comparisons work correctly
print("\n7. DATE VALIDATION")

df["booking_datetime"] = pd.to_datetime(
    df["booking_datetime"], errors="coerce"
)

df["appointment_datetime"] = pd.to_datetime(
    df["appointment_datetime"], errors="coerce"
)

print("Invalid booking dates:",
      df["booking_datetime"].isna().sum())

print("Invalid appointment dates:",
      df["appointment_datetime"].isna().sum())

print("Booking after or at appointment time:",
      (df["booking_datetime"] >=
       df["appointment_datetime"]).sum())


# Check whether lead time contains negative values
print("\n8. LEAD TIME CHECK")
print("Negative lead times:",
      (df["lead_time_days"] < 0).sum())


# Check whether target values are valid binary labels
print("\n9. TARGET VALIDATION")
print("Invalid target values:",
      (~df["no_show"].isin([0, 1])).sum())


# Check age and distance for unrealistic values
print("\n10. RANGE CHECKS")

print("Invalid ages:",
      ((df["age"] < 0) | (df["age"] > 120)).sum())

print("Negative distances:",
      (df["distance_km"] < 0).sum())


# Check whether previous no-shows exceed previous appointments
print("\n11. APPOINTMENT HISTORY CHECK")

print("Invalid history records:",
      (df["prior_no_shows"] >
       df["prior_appointments"]).sum())


# Check whether previous attended visit dates are valid
valid_visit_history = (
    df["days_since_last_visit"].notna()
    & (df["days_since_last_visit"] < 0)
)

print("Negative days since last visit:",
      valid_visit_history.sum())


# List features that must not be used for model training
print("\n12. LEAKAGE REVIEW")

excluded_features = [
    "appointment_id",
    "patient_id",
    "appointment_datetime",
    "no_show",
    "latent"
]

print("Excluded from model features:")
for feature in excluded_features:
    print("-", feature)


# Display final validation message
print("\n" + "=" * 55)
print("DATASET VALIDATION COMPLETED")
print("=" * 55)
