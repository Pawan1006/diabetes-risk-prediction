# 🏥 Diabetes & Health Risk Prediction Dashboard

An interactive **Machine Learning web application** for predicting diabetes risk, visualizing key health indicators, and providing **explainable insights** using SHAP.

---

## 🚀 Features

### 🔮 Risk Prediction
- Predicts diabetes risk with **percentage probability**
- Categorizes users into **Low**, **Medium**, and **High** risk levels
- Provides **personalized lifestyle recommendations**

### 📊 Explainable AI (SHAP)
- Shows which user inputs **increase** or **decrease** diabetes risk
- Color-coded SHAP bar charts for easy interpretation
- Helps users understand *why* the model made a prediction

### 📈 Interactive Analytics Dashboard
- Diabetes vs Non-Diabetes distribution
- BMI comparison by diabetes status
- Top correlated features
- Confusion matrix & ROC curve visualizations
- Random sample predictions

### 🌐 Streamlit App
- Clean, modern UI
- Fully interactive components
- Ready for deployment

---

## 📂 Dataset

Dataset from the **CDC BRFSS 2015** survey (balanced version).

📥 **Kaggle Link:**  
https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

Dataset includes:
- Lifestyle habits  
- Medical history  
- Mental & physical health  
- Demographics (Age group, Income, Education)

---

## 🧠 Machine Learning Pipeline

The app uses a Scikit-learn Pipeline:

1. **Preprocessing**
   - Median & mode imputation  
   - Standard scaling (numeric features)

2. **Feature Selection**
   - `SelectKBest(mutual_info_classif)`  
   - Keeps the **top 8** most informative predictors  

3. **Model**
   - Random Forest Classifier  
   - Tuned for balanced accuracy and interpretability  

4. **Explainability**
   - SHAP Explainer with a wrapped predict function  
   - Ensures compatibility with Scikit-learn pipelines  

---

## 📦 Tech Stack

| Component       | Technology |
|----------------|------------|
| Language       | Python     |
| ML Framework   | Scikit-learn |
| Explainability | SHAP       |
| UI Framework   | Streamlit  |
| Visualization  | Plotly, Matplotlib, Seaborn |
| Model Storage  | Joblib     |

---

## ▶ Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

---

📦 Diabetes-Prediction-App
 ┣ 📂 data/
 ┃ ┗ diabetes_binary_5050split_health_indicators_BRFSS2015.csv
 ┣ 📂 models/
 ┃ ┗ diabetes_model.pkl
 ┣ 📜 app.py
 ┣ 📜 train_model.py
 ┣ 📜 requirements.txt
 ┗ 📜 README.md

---

🙌 Credits
Dataset provided by CDC BRFSS 2015.
Balanced dataset curated by Alex Teboul (Kaggle).

---

⭐ If you like this project…
Feel free to ⭐ star the repository or fork it for improvements!