import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, classification_report, ConfusionMatrixDisplay
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
import pickle
import streamlit as st
import os

# Load dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    df.rename(str.strip, axis='columns', inplace=True)  # Strip unnecessary spaces in column names
    return df

# Clean dataset
def clean_data(df):
    df.replace({'yes': 1, 'no': 0}, inplace=True)  # Encode categorical data
    return df

# Exploratory Data Analysis (EDA)
def eda(df):
    print(df.describe().T)
    print(df.info())

    # Rainfall distribution
    plt.pie(df['rainfall'].value_counts().values,
            labels=df['rainfall'].value_counts().index,
            autopct="%1.1f%%")
    plt.title('Rainfall Distribution')
    plt.show()

    # Numerical features and outlier detection
    features = list(df.select_dtypes(include=np.number).columns)
    if 'day' in features:
        features.remove('day')  # Exclude day column
    print(f"Numerical Features: {features}")

    plt.subplots(figsize=(15, 8))
    for i, col in enumerate(features):
        plt.subplot(3, 4, i + 1)
        sns.boxplot(df[col])
    plt.tight_layout()
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(10, 10))
    sns.heatmap(df.corr(), annot=True, cbar=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()

# Preprocess and split data
def preprocess_and_split(df):
    # Drop irrelevant features
    df.drop(['maxtemp', 'mintemp'], axis=1, inplace=True)

    # Separate features and target
    features = df.drop(['day', 'rainfall'], axis=1)
    target = df['rainfall']

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    features = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)

    # Split the data
    X_train, X_val, Y_train, Y_val = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=2
    )

    # Balance the dataset
    ros = RandomOverSampler(sampling_strategy='minority', random_state=22)
    X_train, Y_train = ros.fit_resample(X_train, Y_train)

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    return X_train, X_val, Y_train, Y_val, scaler, imputer

# Train and evaluate models
def train_and_evaluate(X_train, X_val, Y_train, Y_val):
    models = [
        LogisticRegression(),
        XGBClassifier(),
        SVC(kernel='rbf', probability=True)
    ]

    best_model = None
    best_auc = 0

    for model in models:
        model.fit(X_train, Y_train)
        train_preds = model.predict_proba(X_train)[:, 1]
        val_preds = model.predict_proba(X_val)[:, 1]

        print(f"{model.__class__.__name__}:")
        train_auc = roc_auc_score(Y_train, train_preds)
        val_auc = roc_auc_score(Y_val, val_preds)
        print(f"Training AUC: {train_auc:.3f}")
        print(f"Validation AUC: {val_auc:.3f}\n")

        if val_auc > best_auc:
            best_auc = val_auc
            best_model = model

    ConfusionMatrixDisplay.from_estimator(best_model, X_val, Y_val)
    plt.title("Confusion Matrix")
    plt.show()

    print("Classification Report:")
    print(classification_report(Y_val, best_model.predict(X_val)))

    return best_model

# AI pipeline
def ai_pipeline(filepath):
    df = load_data(filepath)
    df = clean_data(df)
    eda(df)
    X_train, X_val, Y_train, Y_val, scaler, imputer = preprocess_and_split(df)
    best_model = train_and_evaluate(X_train, X_val, Y_train, Y_val)
    return best_model, scaler, imputer

# Run the pipeline
filepath = "c:/Users/F4in/Downloads/Rainfall.csv"
model, scaler, imputer = ai_pipeline(filepath)

# Save Model, Scaler, and Imputer
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

with open('imputer.pkl', 'wb') as file:
    pickle.dump(imputer, file)

print("Model, scaler, and imputer saved successfully.")

# Streamlit App
FEATURE_NAMES = ['pressure', 'temperature', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

st.title("Rainfall Prediction App")

# Collect user inputs
pressure = st.number_input("Enter Pressure (hPa): ")
temperature = st.number_input("Enter Temperature (°C): ")
dewpoint = st.number_input("Enter Dewpoint: ")
humidity = st.number_input("Enter Humidity (%): ")
cloud = st.number_input("Enter Cloud Cover: ")
sunshine = st.number_input("Enter Sunshine: ")
winddirection = st.number_input("Enter Wind Direction: ")
windspeed = st.number_input("Enter Wind Speed (km/ph): ")

if st.button("Predict"):
    # Load model, scaler, and imputer
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    with open('imputer.pkl', 'rb') as imputer_file:
        imputer = pickle.load(imputer_file)

    # Create a DataFrame for the input features
    input_dict = {
        'pressure': [pressure],
        'temperature': [temperature],
        'dewpoint': [dewpoint],
        'humidity': [humidity],
        'cloud': [cloud],
        'sunshine': [sunshine],
        'winddirection': [winddirection],
        'windspeed': [windspeed]
    }
    input_features = pd.DataFrame(input_dict)

    # Impute and scale features
    input_features = pd.DataFrame(imputer.transform(input_features), columns=FEATURE_NAMES)
    scaled_features = scaler.transform(input_features)

    # Predict probabilities
    probabilities = model.predict_proba(scaled_features)[0]
    rain_prob = probabilities[1] * 100
    no_rain_prob = probabilities[0] * 100

    st.write(f"Probability of Rain: {rain_prob:.2f}%")
    st.write(f"Probability of No Rain: {no_rain_prob:.2f}%")