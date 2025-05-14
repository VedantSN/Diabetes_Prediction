# 🩺 Diabetes Prediction Web App using SVM

This is a simple and interactive **Streamlit** web application for predicting whether a person is diabetic or not based on diagnostic measurements. The model is trained using the **Support Vector Machine (SVM)** algorithm with a **linear kernel**, and evaluated using a well-known dataset: `diabetes.csv`.

---

## 📊 Features

- View dataset and its outcome distribution
- Visualizations:
  - Outcome value counts
  - Mean values for each class
  - Confusion matrix heatmap
- Accuracy scores for training and testing sets
- Classification report (precision, recall, f1-score)
- Custom input prediction for diabetes
- Clean and responsive UI with **Streamlit**

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/diabetes-prediction-streamlit.git
cd diabetes-prediction-streamlit
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
streamlit run app.py
