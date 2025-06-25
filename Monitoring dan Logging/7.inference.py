import requests
import joblib
import json
import pandas as pd

def prediction(data):
    url = 'http://127.0.0.1:5002/invocations'
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, data=data)
    response = response.json().get('predictions', [])
    return response

columns =  ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

data = [2, 84, 0, 0, 0, 0.0, 0.304, 21]

df = pd.DataFrame([data], columns=columns)

json_output = {
    "dataframe_split": {
        "columns": df.columns.tolist(),
        "data": df.values.tolist()
    }
}

data_testing = json.dumps(json_output)
print(prediction(data_testing))

