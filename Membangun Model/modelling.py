import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os

mlflow.set_tracking_uri("http://127.0.0.1:5000/")

mlflow.set_experiment("Diabetes Prediction Experiment")

# Mengubah nama pengguna sesuai dengan ID Dicoding
os.environ['MLFLOW_TRACKING_USERNAME'] = 'Rouf Aufalin'



data = pd.read_csv("diabetes.csv")

X_train, X_test, y_train, y_test = train_test_split(
    data.drop("Outcome", axis=1),
    data["Outcome"],
    test_size=0.2,
    random_state=42
)

input_example = X_train[0:5]

with mlflow.start_run():
    #log parameters
    n_estimators = 505
    max_depth = 10
    mlflow.autolog()

    # Train the model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example,
        
    )

    model.fit(X_train, y_train)

    #Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
