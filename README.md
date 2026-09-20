# Pricing Analytics Decision Lab - Version 4.0

This version changes the lab from a slider-based sandbox into a guided **prediction -> experiment -> decision -> optimization -> feedback** activity.

## Student sequence

1. Predict which days should support the highest prices from the logit demand parameters.
2. Build and lock a seven-day theme-park pricing strategy.
3. Benchmark the strategy against optimized variable prices.
4. Predict the direction and value of the price change before responding to a Saturday capacity shock.
5. Predict the direction and value of the price change before responding to a Tuesday demand shock.
6. Identify price sensitivity and predict the price relationship before testing segmented pricing.
7. Generate a verified PDF report.

The app does **not** immediately mark the prediction MCQs correct or incorrect. Students commit to their prediction first, make the pricing decision, and only then see the model result.

## Verified report

The final PDF records:
- student name and ID;
- exact generation timestamp in U.S. Eastern Time and UTC;
- a 12-character SHA-256 verification code based on student ID and timestamp;
- all pre-analysis predictions;
- student pricing decisions;
- optimized benchmark results.

## Files to upload to GitHub

```text
app.py
requirements.txt
README.md
.streamlit/config.toml
data/theme_park_parameters.csv
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Important instructor note

The verification code is useful for detecting duplicate generated reports. It is not cryptographic proof that a PDF has never been edited after generation.
