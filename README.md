# false-decline-aware-fraud-model
Fraud Detection: False-Decline Reduction Model For Payment Authorizations

A fraud detection system for card payment authorizations, built to minimize **total business cost** — not just maximize fraud detection. Optimizes the tradeoff between missed fraud and false declines, since industry research shows false declines cost merchants more than the fraud they're meant to prevent.

## Key Results
- **87.1%** of fraud caught on a fully held-out test set
- **5.1%** reduction in total business cost vs. a default 0.5 threshold
- Threshold selected on a validation set and applied once to test — no data leakage
- Three-tier decisioning (Approve / Review / Decline) instead of a binary flag, avoiding over-declining borderline transactions

## Approach
1. **EDA** — identified fraud clusters by merchant category (up to 16x baseline) and time of day (up to 25x baseline)
2. **Feature engineering** — card-specific spending deviation (rolling window), cyclical time encoding, transaction velocity
3. **Modeling** — XGBoost with early stopping and class-imbalance handling, benchmarked against a Logistic Regression baseline
4. **Business impact** — custom cost matrix (missed fraud vs. false decline), threshold optimized on validation, evaluated once on test
5. **Deployment** — live Streamlit app simulating real-time authorization, including a mock feature store for card history lookup

## Tech Stack
Python · Pandas · NumPy · Scikit-learn · XGBoost · Streamlit · SQL (feature engineering via window functions)

## Project Structure
├── app.py # Streamlit app
├── false_decline_aware_fraud_model.ipynb # Full analysis notebook
├── fraud_model.json # Trained XGBoost model
├── feature_columns.json # Model input schema
├── feature_store.json # Mock feature store (card history)
└── requirements.txt
## Running Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
## Data
Trained on 200,000 synthetic transactions generated via [Sparkov](https://github.com/namebrandon/Sparkov_Data_Generation).
## Limitations
- Cold-start predictions (new cards with no history) are less stable than predictions for cards with established transaction patterns
- Probability outputs are not fully calibrated — strong at ranking risk, less precise as literal probabilities
- Synthetic data; results are directional, not a claim about real-world fraud rates
