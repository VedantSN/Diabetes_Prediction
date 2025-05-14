import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
diabetes_data = pd.read_csv("diabetes.csv")

# Preprocessing
X = diabetes_data.drop(columns="Outcome", axis=1)
y = diabetes_data["Outcome"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=2, stratify=y
)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Model training
model = svm.SVC(kernel="linear")
model.fit(X_train, Y_train)

# Evaluation
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)
train_acc = accuracy_score(Y_train, train_preds)
test_acc = accuracy_score(Y_test, test_preds)

# Streamlit UI
st.title("Diabetes Prediction using SVM")
st.write("## Dataset Preview")
st.dataframe(diabetes_data.head())

st.write("### Distribution of Outcome")
st.bar_chart(diabetes_data["Outcome"].value_counts())

st.write("### Outcome-wise Mean Values")
st.dataframe(diabetes_data.groupby("Outcome").mean())

st.write("### Model Accuracy")
st.write(f"Training Accuracy: {train_acc:.2f}")
st.write(f"Testing Accuracy: {test_acc:.2f}")

st.write("### Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(Y_test, test_preds), annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

st.write("### Classification Report")
st.text(classification_report(Y_test, test_preds))

st.write("## Predict for Custom Input")
input_data = []
feature_names = X.columns.tolist()

for col in feature_names:
    value = st.number_input(f"Enter value for {col}", value=float(diabetes_data[col].mean()))
    input_data.append(value)

if st.button("Predict"):
    input_array = np.asarray(input_data).reshape(1, -1)
    input_array = scaler.transform(input_array)
    prediction = model.predict(input_array)
    result = "Patient is diabetic" if prediction[0] == 1 else "Patient is non-diabetic"
    st.success(result)
