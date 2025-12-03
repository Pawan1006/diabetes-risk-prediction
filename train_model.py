# train_model.py
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# -------------------------
# 1. Load dataset
# -------------------------
df = pd.read_csv("data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
df.replace(" ", np.nan, inplace=True)

TARGET = "Diabetes_binary"
y = df[TARGET]
X = df.drop(columns=[TARGET])

# -------------------------
# 2. Identify features
# -------------------------
numeric_features = ["BMI", "MentHlth", "PhysHlth", "Age"]
categorical_features = [c for c in X.columns if c not in numeric_features]

# -------------------------
# 3. Preprocess
# -------------------------
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# -------------------------
# 4. Feature Selection + Model
# -------------------------
feature_selector = SelectKBest(mutual_info_classif, k=min(8, X.shape[1]))

rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)

clf = Pipeline([
    ("preprocessor", preprocessor),
    ("feature_selection", feature_selector),
    ("model", rf),
])

# -------------------------
# 5. Train & Evaluate
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("Validation Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Validation F1:", round(f1_score(y_test, y_pred), 4))
print("Validation AUC:", round(roc_auc_score(y_test, y_proba), 4))

# -------------------------
# 6. Save final model + metadata
# -------------------------
os.makedirs("models", exist_ok=True)

defaults = {}
for c in X.columns:
    defaults[c] = X[c].median() if c in numeric_features else X[c].mode().iat[0]

save_obj = {
    "pipeline": clf,
    "feature_names": list(X.columns),
    "defaults": defaults
}

joblib.dump(save_obj, "models/diabetes_model.pkl")
print("Saved model → models/diabetes_model.pkl")