
import pandas as pd
from pathlib import Path


# Load the raw dataset so the original data remains unchanged
df = pd.read_csv("data/raw/hospital_appointments.csv")

print("Original dataset shape:", df.shape)


# Convert date columns so records can be sorted by booking time
df["booking_datetime"] = pd.to_datetime(df["booking_datetime"])
df["appointment_datetime"] = pd.to_datetime(df["appointment_datetime"])


# Remove duplicate appointment IDs so one appointment is counted only once
df = df.drop_duplicates(subset="appointment_id", keep="first")


# Remove records with invalid dates or missing target values
df = df.dropna(
    subset=["booking_datetime", "appointment_datetime", "no_show"]
)


# Remove appointments booked after or at their scheduled time
df = df[
    df["booking_datetime"] < df["appointment_datetime"]
]


# Create an indicator because missing last-visit dates can mean no attended visit
df["has_previous_attended_visit"] = (
    df["days_since_last_visit"].notna().astype(int)
)


# Sort by booking time to prevent future records entering earlier datasets
df = df.sort_values("booking_datetime").reset_index(drop=True)


# Calculate chronological split positions
total_records = len(df)
train_end = int(total_records * 0.70)
validation_end = int(total_records * 0.85)


# Create training, validation and testing datasets in time order
train_data = df.iloc[:train_end].copy()
validation_data = df.iloc[train_end:validation_end].copy()
test_data = df.iloc[validation_end:].copy()


# Create the processed folder if it does not already exist
Path("data/processed").mkdir(parents=True, exist_ok=True)


# Save each dataset separately so model training and evaluation use fixed splits
train_data.to_csv("data/processed/train.csv", index=False)
validation_data.to_csv("data/processed/validation.csv", index=False)
test_data.to_csv("data/processed/test.csv", index=False)


# Display dataset sizes to verify the chronological split
print("\nPREPROCESSING COMPLETED")

print("Original records:", total_records)
print("Training records:", len(train_data))
print("Validation records:", len(validation_data))
print("Testing records:", len(test_data))


# Display date ranges to verify that the splits are chronological
print("\nDATE RANGES")

print("Training:",
      train_data["booking_datetime"].min(),
      "to",
      train_data["booking_datetime"].max())

print("Validation:",
      validation_data["booking_datetime"].min(),
      "to",
      validation_data["booking_datetime"].max())

print("Testing:",
      test_data["booking_datetime"].min(),
      "to",
      test_data["booking_datetime"].max())


# Confirm that the three datasets contain no overlapping booking dates
print("\nCHRONOLOGICAL CHECK")

print(
    "Training before validation:",
    train_data["booking_datetime"].max()
    <= validation_data["booking_datetime"].min()
)

print(
    "Validation before testing:",
    validation_data["booking_datetime"].max()
    <= test_data["booking_datetime"].min()
)

print("\nProcessed files saved successfully!")
