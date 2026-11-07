"""
Synthetic hospital appointment no-show dataset generator.
- Fully synthetic: no real patients, no diagnosis/clinical variables.
- reminder_sms_planned is a BOOKING-TIME POLICY flag (set automatically when the slot is booked,
  not chosen later from model output).
- is_holiday_week is RANDOMLY simulated (not a real holiday calendar).
- weather_risk_month is a SEASONAL PROXY (Jul/Aug/Dec flag), not real weather data.
- History features (prior_appointments, prior_no_shows, days_since_last_visit) use only
  appointments whose scheduled time was already in the past at booking time;
  days_since_last_visit counts only ATTENDED visits.
- Prediction time = moment the appointment is BOOKED. Every feature is known then.
- Target: no_show = 1 if the patient did not attend the appointment.
Run:  python generate_dataset.py --n 6000 --seed 42 --out hospital_appointments.csv
"""
import argparse
import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate(n=6000, seed=42):
    rng = np.random.default_rng(seed)
    n_patients = int(n / 3.2)

    # ---- patient-level (stable) attributes ----
    pat = pd.DataFrame({
        "patient_id": [f"P{100000 + i}" for i in range(n_patients)],
        "age": np.clip(rng.gamma(4.0, 10.5, n_patients).round(), 1, 95).astype(int),
        "gender": rng.choice(["F", "M"], n_patients, p=[0.54, 0.46]),
        "distance_km": np.clip(rng.lognormal(1.8, 0.8, n_patients), 0.5, 60).round(1),
        "payment_type": rng.choice(["Insurance", "Government Scheme", "Self-pay"],
                                   n_patients, p=[0.40, 0.35, 0.25]),
        "latent": rng.normal(0, 0.6, n_patients),  # hidden "reliability" trait
    })

    # ---- appointment-level ----
    pid = rng.choice(n_patients, n, p=rng.dirichlet(np.ones(n_patients) * 2))
    df = pat.iloc[pid].reset_index(drop=True)

    start, end = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-06-30")
    days = (end - start).days
    appt_date = start + pd.to_timedelta(rng.integers(0, days, n), unit="D")
    # clinics closed on Sundays
    appt_date = pd.Series(appt_date)
    appt_date = appt_date.where(appt_date.dt.weekday != 6, appt_date + pd.Timedelta(days=1))
    hour = rng.choice(np.arange(8, 18), n,
                      p=np.array([.07, .13, .14, .12, .06, .08, .12, .12, .10, .06]))
    minute = rng.choice([0, 15, 30, 45], n)
    appt_dt = appt_date + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(minute, unit="m")

    appt_type = rng.choice(["New", "Follow-up", "Routine Check"], n, p=[0.30, 0.45, 0.25])
    lead = np.where(appt_type == "Follow-up",
                    rng.gamma(2.0, 6, n), rng.gamma(1.8, 8, n)).round().astype(int)
    lead = np.clip(lead, 0, 90)
    booking_dt = appt_dt - pd.to_timedelta(lead, unit="D") - pd.to_timedelta(rng.integers(1, 8, n), unit="h")

    df["booking_datetime"] = booking_dt
    df["appointment_datetime"] = appt_dt
    df["lead_time_days"] = (appt_dt.dt.normalize() - booking_dt.dt.normalize()).dt.days.values
    df["appointment_weekday"] = appt_dt.dt.day_name().values
    df["appointment_hour"] = hour
    df["appointment_month"] = appt_dt.dt.month.values
    df["department"] = rng.choice(
        ["General Medicine", "Pediatrics", "Orthopedics", "Dermatology", "ENT", "Gynecology", "Ophthalmology"],
        n, p=[.28, .15, .12, .12, .10, .13, .10])
    df.loc[(df.age < 14), "department"] = rng.choice(["Pediatrics", "General Medicine", "ENT"], (df.age < 14).sum(), p=[.7, .2, .1])
    df.loc[(df.department == "Gynecology") & (df.gender == "M"), "department"] = "General Medicine"
    df["appointment_type"] = appt_type
    df["booking_channel"] = rng.choice(["Phone", "Online Portal", "Mobile App", "Walk-in"],
                                       n, p=[.35, .25, .22, .18])
    df["reminder_sms_planned"] = rng.choice([0, 1], n, p=[0.30, 0.70])
    df["is_holiday_week"] = rng.choice([0, 1], n, p=[0.93, 0.07])
    df["weather_risk_month"] = df["appointment_month"].isin([7, 8, 12]).astype(int)  # monsoon/winter-fog proxy

    # ---- history features: only appointments that were already finished at booking time ----
    df = df.sort_values("booking_datetime").reset_index(drop=True)
    df["appointment_id"] = [f"A{500000 + i}" for i in range(n)]
    wd = df.appointment_weekday.map({"Monday": .20, "Tuesday": 0, "Wednesday": -.05, "Thursday": 0,
                                     "Friday": .25, "Saturday": .35, "Sunday": 0})
    dept = df.department.map({"General Medicine": 0, "Pediatrics": -.25, "Orthopedics": -.10,
                              "Dermatology": .30, "ENT": .10, "Gynecology": -.10, "Ophthalmology": 0})
    typ = df.appointment_type.map({"New": .25, "Follow-up": -.20, "Routine Check": .15})
    chan = df.booking_channel.map({"Phone": 0, "Online Portal": .20, "Mobile App": .15, "Walk-in": -.55})
    pay = df.payment_type.map({"Insurance": -.10, "Government Scheme": .05, "Self-pay": .20})
    age_eff = np.where(df.age < 14, -.15, np.where(df.age < 25, .30, np.where(df.age > 65, -.10, 0)))
    hour_eff = np.where(df.appointment_hour <= 9, .10, np.where(df.appointment_hour >= 16, .20, 0))

    hist = {}
    prior_n, prior_ns, last_visit = [], [], []
    noshow = np.zeros(n, dtype=int)
    # linear predictor without history; history effect is added per-row below
    lin_base = (-2.45 + 0.028 * np.minimum(df.lead_time_days, 45) + wd + dept + typ + chan + pay
                + age_eff + hour_eff + 0.018 * np.minimum(df.distance_km, 30)
                - 0.55 * df.reminder_sms_planned + 0.30 * df.is_holiday_week
                + 0.15 * df.weather_risk_month + df.latent)
    # resolve in booking order but history only counts visits completed before booking
    for i in df.index:
        p = df.at[i, "patient_id"]
        bk = df.at[i, "booking_datetime"]
        # e[0] = SCHEDULED time of the earlier appointment. Keeping only e[0] < bk means the visit
        # had already taken place (outcome known) when this one was booked. Appointments that were
        # booked earlier but are still in the future at bk are excluded.
        past = [e for e in hist.get(p, []) if e[0] < bk]
        pn, pns = len(past), sum(e[1] for e in past)
        attended = [e for e in past if e[1] == 0]  # only visits the patient actually attended
        lv = (bk - max(e[0] for e in attended)).days if attended else np.nan
        prior_n.append(pn); prior_ns.append(pns); last_visit.append(lv)
        rate = pns / pn if pn else 0.0
        z = lin_base[i] + 1.6 * rate + (0.25 if pn == 0 else 0)
        y = int(rng.random() < sigmoid(z))
        noshow[i] = y
        hist.setdefault(p, []).append((df.at[i, "appointment_datetime"], y))

    df["prior_appointments"] = prior_n
    df["prior_no_shows"] = prior_ns
    df["days_since_last_visit"] = last_visit
    df["no_show"] = noshow

    # ---- realistic data-quality issues (documented, for EDA/cleaning) ----
    m = rng.random(n)
    df.loc[m < 0.03, "distance_km"] = np.nan
    df.loc[(m >= 0.03) & (m < 0.045), "age"] = np.nan
    df.loc[(m >= 0.045) & (m < 0.06), "payment_type"] = np.nan
    out_idx = rng.choice(n, 12, replace=False)
    df.loc[out_idx, "distance_km"] = rng.integers(150, 400, 12)  # implausible outliers
    dup = df.sample(15, random_state=seed)  # a few exact duplicate rows
    df = pd.concat([df, dup], ignore_index=True).sort_values("booking_datetime").reset_index(drop=True)

    cols = ["appointment_id", "patient_id", "booking_datetime", "appointment_datetime",
            "lead_time_days", "appointment_weekday", "appointment_hour", "appointment_month",
            "department", "appointment_type", "booking_channel", "reminder_sms_planned",
            "is_holiday_week", "weather_risk_month", "age", "gender", "distance_km",
            "payment_type", "prior_appointments", "prior_no_shows", "days_since_last_visit",
            "no_show"]
    return df[cols]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="hospital_appointments.csv")
    a = ap.parse_args()
    d = generate(a.n, a.seed)
    d.to_csv(a.out, index=False)
    print(d.shape, "no-show rate: %.3f" % d.no_show.mean())
