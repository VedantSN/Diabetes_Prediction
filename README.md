# 🩺 Diabetes Prediction Web App using SVM

This is a simple and interactive **Streamlit** web application for predicting whether a person is diabetic or not based on diagnostic measurements. The model is trained using the **Support Vector Machine (SVM)** algorithm with a **linear kernel**, and evaluated using a well-known dataset: `diabetes.csv`.

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

## 🚀 How to Run the Project

1. Clone the Repository  
```bash
git clone https://github.com/yourusername/diabetes-prediction-streamlit.git
cd diabetes-prediction-streamlit
```

2. Install Requirements  
Use a virtual environment or install directly:  
```bash
pip install -r requirements.txt
```
Or manually:  
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
```

3. Run the App  
```bash
streamlit run app.py
```
Then open the local URL provided in your terminal in a web browser.

## 📁 Project Structure
```
diabetes-prediction-streamlit/
├── diabetes.csv               # Dataset used for training
├── app.py                     # Main Streamlit app
├── README.md                  # Project overview
└── requirements.txt           # Python dependencies
```

## 📈 Model Details
- **Algorithm**: Support Vector Machine (SVM)
- **Kernel**: Linear
- **Scaler**: StandardScaler (for feature normalization)
- **Train/Test Split**: 80/20 stratified split
- **Metrics**: Accuracy, confusion matrix, classification report

## 🧪 Dataset Info
The dataset `diabetes.csv` is sourced from the [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database), which includes the following features:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (0 = Non-diabetic, 1 = Diabetic)


## 📄 License
This project is licensed under the **MIT License**. Feel free to use, modify, and share it.
