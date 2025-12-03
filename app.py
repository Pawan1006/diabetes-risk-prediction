# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score

st.set_page_config(page_title="Diabetes Risk App", layout="wide")

# -------------------------
# Load model + metadata
# -------------------------
saved = joblib.load("models/diabetes_model.pkl")
model = saved["pipeline"]
feature_names = saved["feature_names"]
defaults = saved["defaults"]

ui_features = ["HighBP", "HighChol", "BMI", "Smoker", "Stroke",
               "HeartDiseaseorAttack", "Age", "GenHlth"]

# -------------------------
# Layout
# -------------------------
st.title("🏥 Diabetes Risk Prediction Dashboard")
page = st.sidebar.radio("Navigation", ["Predict Risk", "Data Dashboard", "Model Performance"])

# -------------------------
# Helper: build a full-row input
# -------------------------
def build_input_row(user_inputs):
    row = defaults.copy()
    row.update(user_inputs)
    return pd.DataFrame([row], columns=feature_names)

# -------------------------
# Predict Risk
# -------------------------
if page == "Predict Risk":
    st.sidebar.header("🔍 Enter Patient Details")

    HighBP = st.sidebar.selectbox("High Blood Pressure", [0, 1])
    HighChol = st.sidebar.selectbox("High Cholesterol", [0, 1])
    BMI = st.sidebar.slider("BMI", 10.0, 60.0, 25.0)
    Smoker = st.sidebar.selectbox("Smoker", [0, 1])
    Stroke = st.sidebar.selectbox("Stroke", [0, 1])
    HeartDisease = st.sidebar.selectbox("Heart Disease/Attack", [0, 1])

    age_mapping = {
        "18–24": 1, "25–29": 2, "30–34": 3, "35–39": 4,
        "40–44": 5, "45–49": 6, "50–54": 7, "55–59": 8,
        "60–64": 9, "65–69": 10, "70–74": 11, "75–79": 12, "80+": 13
    }
    age_display = st.sidebar.selectbox("Age Range", list(age_mapping.keys()))
    Age = age_mapping[age_display]

    GenHlth = st.sidebar.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3)

    if st.sidebar.button("Predict Risk"):
        user_inputs = {
            "HighBP": HighBP,
            "HighChol": HighChol,
            "BMI": BMI,
            "Smoker": Smoker,
            "Stroke": Stroke,
            "HeartDiseaseorAttack": HeartDisease,
            "Age": Age,
            "GenHlth": GenHlth
        }

        X_input = build_input_row(user_inputs)
        proba = float(model.predict_proba(X_input)[0][1]) * 100

        # Output
        if proba < 40:
            risk, emoji = "Low Risk", "✅"
        elif proba < 70:
            risk, emoji = "Medium Risk", "⚠️"
        else:
            risk, emoji = "High Risk", "❌"

        st.subheader("Prediction Result")
        st.markdown(f"### {emoji} {risk} — {proba:.2f}% chance of diabetes")

        # -------------------------
        # SHAP: FINAL FIX (KernelExplainer)
        # -------------------------

        st.write("### 🔍 Feature Contribution (SHAP)")

        try:
            # Load background dataset
            df_full = pd.read_csv("data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
            background_df = df_full[feature_names].sample(50, random_state=42)  # faster

            # Prediction wrapper so SHAP never modifies pipeline
            def predict_fn(x):
                x_df = pd.DataFrame(x, columns=feature_names)
                return model.predict_proba(x_df)

            # Use SHAP's universal explainer
            explainer = shap.Explainer(predict_fn, background_df, feature_names=feature_names)

            # Compute SHAP values
            shap_values = explainer(X_input)

            # Extract SHAP values for diabetes class
            sv = shap_values.values[0, :, 1]

            # Filter SHAP values only for UI features
            ui_mask = [feature_names.index(f) for f in ui_features]

            df_shap = pd.DataFrame({
                "Feature": ui_features,
                "Impact": [sv[i] for i in ui_mask]
            }).sort_values(by="Impact", key=abs, ascending=False)


            # Color bars: red = increases risk, green = decreases risk
            df_shap["Color"] = df_shap["Impact"].apply(lambda x: "#EF553B" if x > 0 else "#00CC96")

            fig = px.bar(
                df_shap.head(10),
                x="Impact",
                y="Feature",
                orientation="h",
                title="Top SHAP Feature Impacts",
                color="Color",
                color_discrete_map="identity"
            )

            st.plotly_chart(fig, use_container_width=True)

            # Add explanation
            st.info("""
            ### 📌 How to Interpret SHAP Impact
            - **Red bars** → Increase predicted diabetes risk for this person.
            - **Green bars** → Decrease predicted diabetes risk.
            - **Longer bars** → Stronger influence on the prediction.
            - SHAP values show how much each feature pushed the model **towards** or **away from** predicting diabetes.
            """)


        except Exception as e:
            st.warning("SHAP explanation unavailable: " + str(e))


# -------------------------
# Data Dashboard
# -------------------------
elif page == "Data Dashboard":
    st.header("📊 Dataset Overview")
    df = pd.read_csv("data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

    st.metric("Total Records", len(df))
    st.metric("Diabetes Cases", df["Diabetes_binary"].sum())

    fig = px.pie(df, names="Diabetes_binary",
                 title="Diabetes vs Non-Diabetes")
    st.plotly_chart(fig)

    avg = df.groupby("Diabetes_binary")["BMI"].mean().reset_index()
    fig2 = px.bar(avg, x="Diabetes_binary", y="BMI", title="Average BMI")
    st.plotly_chart(fig2)

# -------------------------
# Model Performance
# -------------------------
elif page == "Model Performance":
    st.header("📈 Model Performance")

    df = pd.read_csv("data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
    X = df[feature_names]
    y = df["Diabetes_binary"]

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    st.metric("Accuracy", round((y_pred == y).mean(), 3))
    st.metric("F1 Score", round(f1_score(y, y_pred), 3))

    fpr, tpr, _ = roc_curve(y, y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="ROC")
    ax.plot([0,1],[0,1], "--")
    st.pyplot(fig)