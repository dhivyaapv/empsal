import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.title("🧠 Employee Salary Prediction: ML Model Comparison")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### 📄 Data Preview", data.head())

    target_column = st.selectbox("Select the target column (e.g., salary)", data.columns)

    if target_column:
        # Encoding categorical columns
        label_enc = LabelEncoder()
        for col in data.columns:
            if data[col].dtype == 'object' and col != target_column:
                data[col] = label_enc.fit_transform(data[col])

        # Encode target
        y = label_enc.fit_transform(data[target_column])
        X = data.drop(columns=[target_column])

        # Scale features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Models
        models = {
            "KNN": KNeighborsClassifier(),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=500),
            "MLP Classifier": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        }

        st.subheader("🧪 Model Accuracy Scores")

        accuracies = []
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)
            st.write(f"**{name}** accuracy: {acc:.4f}")

        # Plot accuracy comparison
        st.subheader("📊 Accuracy Comparison")
        fig, ax = plt.subplots()
        ax.bar(models.keys(), accuracies, color='teal')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Accuracy')
        ax.set_title('ML Models Comparison')
        for i, acc in enumerate(accuracies):
            ax.text(i, acc + 0.01, f"{acc:.2f}", ha='center')
        st.pyplot(fig)
