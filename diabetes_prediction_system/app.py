from flask import Flask, render_template, request
import joblib
import numpy as np
from pathlib import Path

app = Flask(__name__)

# --------------------------------
# 1. CHECK MODEL FILES
# --------------------------------

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


# --------------------------------
# 2. LOAD TRAINED MODEL
# --------------------------------

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("class_encoder.pkl")
feature_names = joblib.load("feature_names.pkl")


# --------------------------------
# 3. INPUT FIELDS
# --------------------------------

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


# --------------------------------
# 4. HOME PAGE
# --------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html",
        fields=FIELDS,
        result=None,
        form_data={}
    )


# --------------------------------
# 5. PREDICTION ROUTE
# --------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    form_data = request.form.to_dict()

    try:

        values = {}

        # Read and validate user inputs
        for key, label, kind in FIELDS:

            raw = form_data.get(key, "").strip()

            if not raw:
                raise ValueError(
                    f"Please enter {label}"
                )

            value = float(raw)

            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    f"Invalid value for {label}"
                )

            # Store using uppercase feature names
            values[key.strip().upper()] = value

        # Validate Age
        if not 1 <= values["AGE"] <= 120:
            raise ValueError(
                "Age must be between 1 and 120."
            )

        # Validate BMI
        if values["BMI"] <= 0:
            raise ValueError(
                "BMI must be greater than zero."
            )

        # --------------------------------
        # 6. FEATURE ORDERING
        # --------------------------------

        normalized_values = {
            key.strip().upper(): value
            for key, value in values.items()
        }

        normalized_features = [
            str(name).strip().upper()
            for name in feature_names
        ]

        missing_features = [
            name for name in normalized_features
            if name not in normalized_values
        ]

        if missing_features:
            raise ValueError(
                f"Missing model features: {missing_features}. "
                "Check feature_names.pkl against the HTML form."
            )

        input_data = np.array(
            [[normalized_values[name]
              for name in normalized_features]],
            dtype=float
        )

        # --------------------------------
        # 7. FEATURE SCALING
        # --------------------------------

        scaled_data = scaler.transform(input_data)

        # --------------------------------
        # 8. MODEL PREDICTION
        # --------------------------------

        prediction = model.predict(scaled_data)[0]

        probabilities = model.predict_proba(scaled_data)[0]

        # --------------------------------
        # 9. DECODE PREDICTION
        # --------------------------------

        prediction_label = encoder.inverse_transform(
            [prediction]
        )[0]

        confidence = round(
            float(np.max(probabilities)) * 100,
            2
        )

        # --------------------------------
        # 10. DISPLAY RESULT
        # --------------------------------

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

    except (ValueError, KeyError) as exc:

        return render_template(
            "index.html",
            fields=FIELDS,
            result=None,
            form_data=form_data,
            error=str(exc)
        ), 400


# --------------------------------
# 11. RUN FLASK APPLICATION
# --------------------------------

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )