from flask import Flask, render_template, request
import joblib
import numpy as np
from pathlib import Path

app = Flask(__name__)

# Check model files
ARTIFACTS = [
    "model.pkl",
    "scaler.pkl",
    "class_encoder.pkl",
    "feature_names.pkl"
]

if not all(Path(file).exists() for file in ARTIFACTS):
    raise RuntimeError(
        "Model files are missing. Run python train_model.py first."
    )

# Load trained model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("class_encoder.pkl")
feature_names = joblib.load("feature_names.pkl")

FIELDS = [
    ("AGE", "Age", "number"),
    ("Pregnancies", "Pregnancies", "number"),
    ("Glucose", "Glucose Level", "decimal"),
    ("BloodPressure", "Blood Pressure", "decimal"),
    ("SkinThickness", "Skin Thickness", "decimal"),
    ("Insulin", "Insulin Level", "decimal"),
    ("BMI", "Body Mass Index", "decimal"),
    ("DiabetesPedigreeFunction",
     "Diabetes Pedigree Function", "decimal")
]


@app.route("/")
def home():

    return render_template(
        "index.html",
        fields=FIELDS,
        result=None,
        form_data={}
    )


@app.route("/predict", methods=["POST"])
def predict():

    form_data = request.form.to_dict()

    try:

        values = {}

        for key, label, kind in FIELDS:

            raw = form_data.get(key, "").strip()

            if not raw:
                raise ValueError(f"Please enter {label}")

            value = float(raw)

            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid value for {label}")

            values[key] = value

        if not 1 <= values["AGE"] <= 120:
            raise ValueError("Age must be between 1 and 120.")

        if values["BMI"] <= 0:
            raise ValueError("BMI must be greater than zero.")

        # Arrange input according to training feature order
        input_data = np.array([
            [values[name] for name in feature_names]
        ])

        # Scale input
        scaled_data = scaler.transform(input_data)

        # Predict
        prediction = model.predict(scaled_data)[0]

        probabilities = model.predict_proba(scaled_data)[0]

        prediction_label = encoder.inverse_transform(
            [prediction]
        )[0]

        confidence = round(
            float(np.max(probabilities)) * 100, 2
        )

        if prediction_label == 0:

            title = "No Diabetes Detected"
            status = "success"

        else:

            title = "Diabetes Detected"
            status = "danger"

        result = {
            "code": prediction_label,
            "title": title,
            "status": status,
            "confidence": confidence
        }

        return render_template(
            "index.html",
            fields=FIELDS,
            result=result,
            form_data=form_data
        )

    except ValueError as exc:

        return render_template(
            "index.html",
            fields=FIELDS,
            result=None,
            form_data=form_data,
            error=str(exc)
        ), 400


if __name__ == "__main__":

    app.run(debug=True)