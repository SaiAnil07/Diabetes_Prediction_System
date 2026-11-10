# GlucoGuide — Diabetes Prediction System

A Flask + Scikit-learn web app based on the supplied project report. The ML architecture is a Random Forest classifier and the dataset target uses three classes: `N`, `P`, and `Y`.

## Project structure

```text
diabetes_prediction_redesigned/
├── app.py
├── train_model.py
├── requirements.txt
├── Dataset of Diabetes.csv   # Add your dataset here
├── model.pkl                 # Generated after training
├── scaler.pkl                # Generated after training
├── class_encoder.pkl         # Generated after training
├── feature_names.pkl         # Generated after training
├── templates/
│   └── index.html
└── static/
    └── css/
        └── style.css
```

## Setup (Windows)

1. Install Python 3.10 or later.
2. Open a terminal in this folder.
3. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   venv\\Scripts\\activate
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Put `Dataset of Diabetes.csv` in this same folder. The expected columns are:

   `Gender, AGE, Urea, Cr, HbAlc, Chol, TG, HDL, LDL, VLDL, BMI, CLASS`

   Optional ID columns `ID` and `No_Pation` are ignored. `CLASS` must contain `N`, `P`, and/or `Y`; gender should be `M` or `F`.

6. Train and save the model:

   ```bash
   python train_model.py
   ```

7. Start the website:

   ```bash
   python app.py
   ```

8. Open `http://127.0.0.1:5000` in your browser.

## Interface changes

The original single Bootstrap card has been changed to a responsive two-column screening layout, with a teal/navy palette, grouped patient and clinical fields, a redesigned result panel, and probability bars. The model inputs and dataset target classes remain aligned with the supplied report.

## Important

This is an academic demonstration, not a diagnostic medical device. The model's confidence score is not an individual's clinical probability of diabetes. Do not use it to make treatment decisions. A qualified healthcare professional must interpret appropriate laboratory tests.
