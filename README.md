# Pricing Analytics Decision Lab — Version 3.0

This Streamlit app is designed for BAN 517: Supply Chain Analytics.

## Purpose

The lab complements the pricing assignment rather than replacing it.

Students first complete the analytical work in Excel/Python/Solver. The Streamlit lab then gives them an applied pricing experience:

1. Build a seven-day theme-park pricing strategy manually.
2. Lock the strategy.
3. Benchmark it against optimized variable pricing.
4. Respond to a Saturday capacity shock.
5. Respond to a Tuesday demand shock.
6. Compare one-price and segmented-pricing decisions.
7. Generate and download a verified PDF results report for submission.

The app intentionally requires a student decision before revealing the optimized benchmark.

## Theme-park model

The app uses the same logit demand structure as the assignment:

`d(p) = D * exp(a + b*p) / (1 + exp(a + b*p))`

Baseline daily capacity: 1,000 customers  
Marginal cost: $0 per customer

| Day | D | a | b |
|---|---:|---:|---:|
| Sunday | 4500 | 5 | -0.5 |
| Monday | 1500 | 4 | -0.4 |
| Tuesday | 1400 | 3 | -0.3 |
| Wednesday | 1500 | 2 | -0.2 |
| Thursday | 2000 | 3 | -0.3 |
| Friday | 4100 | 4 | -0.4 |
| Saturday | 5300 | 5 | -0.5 |

## Files

- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — appearance settings
- `data/theme_park_parameters.csv` — baseline parameters
- `README.md` — this file

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy with Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload all files in this package.
3. Go to Streamlit Community Cloud.
4. Create a new app from the repository.
5. Set the main file path to `app.py`.
6. Deploy.
7. Add the resulting URL to Canvas.

## Instructor notes

- Students should complete the original pricing assignment calculations separately.
- The app is best used after Questions 1–4.
- The final PDF records the exact server generation timestamp in U.S. Eastern Time and UTC.
- The PDF includes a 12-character SHA-256 verification code generated from the student ID and exact timestamp.
- The same timestamp and verification code are printed in the report and footer.
- A Student ID is required before the final PDF can be generated.
- The timestamp is created when the student clicks **Generate Verified PDF Report** and remains fixed for that generated report.
- The optimizer uses a $0.25 price grid so results are stable and easy for students to compare.
- No generative AI is required for this version of the lab.


## Submission verification

The final report uses Python's `datetime` module with timezone-aware timestamps. The app records the generation time to microsecond precision in both U.S. Eastern Time and UTC.

The verification code is generated as:

```text
SHA-256(student_id | exact_timestamp) -> first 12 hexadecimal characters
```

This is useful for detecting identical submitted reports: two copies of the same generated report will have the same verification code. The code is a duplicate-detection aid, not a cryptographic proof of academic integrity. For stronger tamper resistance, deploy the app with a server-held secret and include it in an HMAC signature.
