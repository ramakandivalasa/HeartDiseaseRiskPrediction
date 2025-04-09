from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mongodb_connect import get_db

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("C:/Users/ramak/OneDrive/Documents/my-express-app/HDP/model.pkl", "rb"))


# Preprocessing function to match training data
def preprocess_input(user_input):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    df = pd.DataFrame([user_input])

    # Add age group feature (as done in your training code)
    age = df["Age"].values[0]
    if age < 35:
        df["AgeGroups"] = 1
    elif age < 45:
        df["AgeGroups"] = 2
    elif age < 55:
        df["AgeGroups"] = 3
    elif age < 65:
        df["AgeGroups"] = 4
    elif age < 75:
        df["AgeGroups"] = 5
    else:
        df["AgeGroups"] = 6

    # Define the columns
    quantitative_variables = ["Age", "MaxHR", "Oldpeak"]
    categorical_variables = ["Sex", "ChestPain", "Fbs", "ExAng", "Slope", "AgeGroups"]

    # Scale quantitative columns
    scaler = StandardScaler()
    df[quantitative_variables] = scaler.fit_transform(df[quantitative_variables])

    # Convert categorical columns to dummies
    df = pd.get_dummies(df, columns=categorical_variables)

    # Drop extra columns you dropped in training
    to_drop = ["Sex_1", "ChestPain_3", "Fbs_1", "ExAng_1", "Slope_3", "AgeGroups_6"]
    for col in to_drop:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Add missing columns that model was trained on
    required_cols = ['Age', 'MaxHR', 'Oldpeak',
                     'Sex_0',
                     'ChestPain_0', 'ChestPain_1', 'ChestPain_2',
                     'Fbs_0',
                     'ExAng_0',
                     'Slope_1', 'Slope_2',
                     'AgeGroups_1', 'AgeGroups_2', 'AgeGroups_3',
                     'AgeGroups_4', 'AgeGroups_5']

    for col in required_cols:
        if col not in df.columns:
            df[col] = 0  # Add missing dummy as 0

    # Reorder columns
    df = df[required_cols]

    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form input
        data = {
            "Age": int(request.form['age']),
            "Sex": int(request.form['sex']),
            "ChestPain": int(request.form['chest_pain']),
            "Fbs": int(request.form['fbs']),
            "ExAng": int(request.form['exang']),
            "Slope": int(request.form['slope']),
            "MaxHR": int(request.form['maxhr']),
            "Oldpeak": float(request.form['oldpeak']),
        }

        # Preprocess input
        input_df = preprocess_input(data)

        # Predict
        prediction = model.predict(input_df)[0]

        # Save to MongoDB
        db = get_db()
        db.insert_one({**data, 'prediction': int(prediction)})

        result = "Positive (At Risk)" if prediction == 1 else "Negative (Low Risk)"
        return render_template('result.html', result=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
