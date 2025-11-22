import os
import pickle
import streamlit as st
import pandas as pd

# ---------------------
# PAGE CONFIG
# ---------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
)

# ---------------------
# LOAD MODEL
# ---------------------
model_path = os.path.join("models", "churn_model.pkl")

with open(model_path, "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
num_imputer = saved["num_imputer"]
num_scaler = saved["num_scaler"]
numeric_cols = saved["numeric_cols"]
model_accuracy = saved.get("accuracy", None)
model_cm = saved.get("confusion_matrix", None)
model_report = saved.get("report", None)

# ---------------------
# CSS – gradient background, nice button, prediction card
# ---------------------
st.markdown(
    """
    <style>
    /* App background (soft gradient) */
    .stApp {
        background: radial-gradient(circle at top left, #e0f2fe, #eef2ff 45%, #f9fafb 100%);
    }

    /* Prediction card with small hover */
    .prediction-card {
        padding: 1.2rem 1.4rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #0f172a, #111827);
        border: 1px solid rgba(148,163,184,0.6);
        color: #e5e7eb;
        box-shadow: 0 14px 35px rgba(15,23,42,0.55);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    .prediction-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 20px 45px rgba(15,23,42,0.7);
    }

    /* Button styling */
    .stButton>button {
        border-radius: 999px;
        padding: 0.5rem 1.6rem;
        font-weight: 600;
        border: none;
        background: linear-gradient(135deg, #22c55e, #16a34a);
        color: white;
        box-shadow: 0 10px 25px rgba(22,163,74,0.45);
        transition: transform 0.1s ease, box-shadow 0.1s ease, filter 0.1s ease;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 34px rgba(22,163,74,0.6);
        filter: brightness(1.03);
    }

    /* Slight floating animation for the title */
    .title-anim {
        animation: floatTitle 3.2s ease-in-out infinite alternate;
    }
    @keyframes floatTitle {
        from { transform: translateY(0px); }
        to   { transform: translateY(-3px); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------
# SIDEBAR
# ---------------------
st.sidebar.title("About the Model")

if model_accuracy is not None:
    st.sidebar.metric("Test Accuracy", f"{model_accuracy:.2%}")

if model_cm is not None:
    st.sidebar.write("Confusion Matrix (test data):")
    cm_df = pd.DataFrame(
        model_cm,
        index=["Actual 0", "Actual 1"],
        columns=["Pred 0", "Pred 1"],
    )
    st.sidebar.dataframe(cm_df)

if model_report is not None:
    with st.sidebar.expander("Classification Report"):
        st.text(model_report)


# ---------------------
# MAIN LAYOUT (no white box)
# ---------------------
col_title, col_desc = st.columns([2, 3])

with col_title:
    st.markdown(
        "<h2 class='title-anim'>📉 Customer Churn Prediction</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#4b5563;'>Estimate the risk of losing a customer based on their behaviour.</p>",
        unsafe_allow_html=True,
    )

with col_desc:
    st.markdown(
        "<p style='font-size:0.9rem;color:#6b7280;'>"
        "Enter customer information and usage statistics. The model analyses tenure, interactions, "
        "spend and support history to predict churn risk."
        "</p>",
        unsafe_allow_html=True,
    )

st.markdown("---")

left, right = st.columns([3, 2])

# ---------------------
# INPUT FORM
# ---------------------
with left:
    st.subheader("Enter customer details")

    values = {}
    for col in numeric_cols:
        label = col
        if "Age" in col:
            values[col] = st.number_input(label, min_value=0.0, value=30.0)
        elif "Tenure" in col:
            values[col] = st.number_input(label, min_value=0.0, value=12.0)
        elif "Usage" in col:
            values[col] = st.number_input(label, min_value=0.0, value=10.0)
        elif "Support" in col:
            values[col] = st.number_input(label, min_value=0.0, value=1.0)
        elif "Delay" in col:
            values[col] = st.number_input(label, min_value=0.0, value=0.0)
        elif "Spend" in col:
            values[col] = st.number_input(label, min_value=0.0, value=500.0)
        elif "Interaction" in col:
            values[col] = st.number_input(label, min_value=0.0, value=7.0)
        else:
            values[col] = st.number_input(label, value=0.0)

HISTORY_FILE = "prediction_history.csv"
prediction_made = False
pred = None
proba = None

with left:
    if st.button("Predict Churn"):
        # Build numeric DataFrame in correct order
        X_num = pd.DataFrame([[values[c] for c in numeric_cols]], columns=numeric_cols)

        # Impute + scale
        X_num_imp = num_imputer.transform(X_num)
        X_num_scaled = num_scaler.transform(X_num_imp)
        X_num_scaled = pd.DataFrame(X_num_scaled, columns=numeric_cols)

        # Final features (numeric only)
        X_final = X_num_scaled

        # Predict
        pred = model.predict(X_final)[0]
        proba = model.predict_proba(X_final)[0][1]
        prediction_made = True

        # Save prediction history
        row = values.copy()
        row["Prediction"] = int(pred)
        row["Churn_Probability"] = float(proba)

        if os.path.exists(HISTORY_FILE):
            hist = pd.read_csv(HISTORY_FILE)
            hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
        else:
            hist = pd.DataFrame([row])

        hist.to_csv(HISTORY_FILE, index=False)
        st.info("Prediction saved to history ✅")

# ---------------------
# PREDICTION CARD
# ---------------------
with right:
    st.subheader("Prediction")

    if prediction_made:
        if pred == 1:
            color = "#f97373"
            text = "Customer is likely to <b>CHURN</b>"
            icon = "⚠️"
        else:
            color = "#22c55e"
            text = "Customer is <b>NOT</b> likely to churn"
            icon = "✅"

        st.markdown(
            f"""
            <div class="prediction-card">
                <h4 style="margin-top:0;color:{color};">{icon} {text}</h4>
                <p style="margin-bottom:0.6rem;">
                    Estimated churn probability: <b>{proba:.2%}</b>
                </p>
                <div style="height:8px;border-radius:999px;background:#1f2937;margin-top:0.4rem;">
                    <div style="height:8px;border-radius:999px;width:{proba*100:.1f}%;background:{color};"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<p style='color:#6b7280;'>Fill in the details on the left and click "
            "<b>Predict Churn</b> to see the result here.</p>",
            unsafe_allow_html=True,
        )

# ---------------------
# HISTORY
# ---------------------
st.subheader("Prediction History")
if os.path.exists(HISTORY_FILE):
    hist = pd.read_csv(HISTORY_FILE)
    st.dataframe(hist)
else:
    st.write("No predictions made yet.")
