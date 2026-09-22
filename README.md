# Pricing Analytics Decision Lab - Version 4.1

This version uses a **hotel revenue-management scenario** that is deliberately different from the graded theme-park data in Questions 3 and 4. The learning sequence remains **prediction -> experiment -> decision -> optimization -> feedback**.

## Student sequence

1. Predict which nights should support the highest room rates from the logit demand parameters.
2. Build and lock a seven-night hotel pricing strategy.
3. Benchmark the strategy against optimized variable room rates.
4. Predict and respond to a Saturday capacity shock when 30 rooms go out of service.
5. Predict and respond to a Tuesday demand shock caused by a convention.
6. Identify price sensitivity and test segmented pricing for business and leisure guests.
7. Generate a verified PDF report.

The app does **not** immediately mark prediction MCQs correct or incorrect. Students commit to a prediction first, make the pricing decision, and only then see the model result.

## Hotel baseline

- Capacity: 120 rooms per night
- Marginal cost used in the lab: $0 for the simplified revenue-management exercise
- Demand model: logit price-response function
- Daily parameters are stored in `data/hotel_parameters.csv`

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
data/hotel_parameters.csv
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Important instructor note

The verification code is useful for detecting duplicate generated reports. It is not cryptographic proof that a PDF has never been edited after generation.
