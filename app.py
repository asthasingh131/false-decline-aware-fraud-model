
import streamlit as st
import pandas as pd
import numpy as np
import json
from xgboost import XGBClassifier

CATEGORIES = [
    "food_dining", "gas_transport", "grocery_net", "grocery_pos", "health_fitness",
    "home", "kids_pets", "misc_net", "misc_pos", "personal_care",
    "shopping_net", "shopping_pos", "travel"
]
DECLINE_THRESHOLD = 0.52
REVIEW_THRESHOLD = 0.30

@st.cache_resource
def load_artifacts():
    model = XGBClassifier()
    model.load_model("fraud_model.json")
    with open("feature_columns.json") as f:
        feature_columns = json.load(f)
    with open("feature_store.json") as f:
        feature_store = pd.DataFrame(json.load(f))
    return model, feature_columns, feature_store

model, feature_columns, feature_store = load_artifacts()

st.title("Fraud Detection - Transaction Risk Scorer")
st.write("Simulates a real-time authorization decision for a card payment network.")

cc_num = st.text_input("Card Number (try a known ID, or any number for a new card)")
amt = st.number_input("Transaction Amount ($)", min_value=0.0, value=50.0)
category = st.selectbox("Merchant Category", CATEGORIES)
hour = st.slider("Transaction Hour (0-23)", 0, 23, 12)
city_pop = st.number_input("Cardholder City Population", min_value=0, value=50000)

if st.button("Score Transaction"):
    card_row = feature_store[feature_store["cc_num"].astype(str) == cc_num]
    if len(card_row) > 0:
        avg_amt_prior_5 = card_row["avg_amt_prior_5"].values[0]
        cumulative_txn_count = card_row["cumulative_txn_count"].values[0]
        st.info("Known card - using stored transaction history.")
    else:
        avg_amt_prior_5 = amt
        cumulative_txn_count = 0
        st.info("New card - no history found, using cold-start defaults.")

    row = {col: 0 for col in feature_columns}
    row["amt"] = amt
    row["cumulative_txn_count"] = cumulative_txn_count
    row["avg_amt_prior_5"] = avg_amt_prior_5
    row["amt_deviation"] = amt - avg_amt_prior_5
    row["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    row["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    row["city_pop"] = city_pop
    cat_col = f"category_{category}"
    if cat_col in row:
        row[cat_col] = 1

    X = pd.DataFrame([row])[feature_columns]
    prob = model.predict_proba(X)[0][1]

    st.metric("Fraud Probability", f"{prob:.2%}")

    if prob >= DECLINE_THRESHOLD:
        st.error(f"DECLINE - probability ({prob:.2%}) exceeds the cost-optimized threshold ({DECLINE_THRESHOLD:.0%})")
    elif prob >= REVIEW_THRESHOLD:
        st.warning(f"REVIEW - probability ({prob:.2%}) is elevated; route for step-up verification")
    else:
        st.success(f"APPROVE - probability ({prob:.2%}) is low risk")

    st.caption(f"Deviation from card's typical spend: ${row['amt_deviation']:.2f}")
