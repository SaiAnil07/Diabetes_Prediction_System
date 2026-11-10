import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "diabetes.csv"

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

TARGET = "Outcome"


def main():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully!")
    print(df.head())

    # Select required columns
    df = df[FEATURES + [TARGET]].copy()

    # Handle missing and invalid values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

        # Treat zero values in these measurements as missing
        if col in ["Glucose", "BloodPressure",
                   "SkinThickness", "Insulin", "BMI"]:
            df[col] = df[col].replace(0, np.nan)

        df[col] = df[col].fillna(df[col].median())

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df.dropna(subset=[TARGET], inplace=True)

    print("Preprocessing completed!")

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    # Encode target
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Training Random Forest Model...")

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    # Evaluation
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nModel Evaluation")
    print("----------------------")
    print("Accuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # Save model files
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(encoder, "class_encoder.pkl")
    joblib.dump(FEATURES, "feature_names.pkl")

    print("\nModel training completed successfully!")

    print("Generated files:")
    print("1. model.pkl")
    print("2. scaler.pkl")
    print("3. class_encoder.pkl")
    print("4. feature_names.pkl")


if __name__ == "__main__":
    main()