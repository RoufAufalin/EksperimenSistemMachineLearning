import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
import numpy as np
import os

mlflow.set_tracking_uri("http://127.0.0.1:5000/")

mlflow.set_experiment("Diabetes Prediction Experiment")

# Mengubah nama pengguna sesuai dengan ID Dicoding.
os.environ['MLFLOW_TRACKING_USERNAME'] = 'Rouf Aufalin'

mlflow.set_experiment("Diabetes Prediction Experiment")
data = pd.read_csv("diabetes.csv")
X_train, X_test, y_train, y_test = train_test_split(
    data.drop("Outcome", axis=1),
    data["Outcome"],
    test_size=0.2,
    random_state=42
)

input_example = X_train[0:5]

# Mendefinisikan Metode Random Search
n_estimators_range = np.linspace(10, 1000, 5, dtype=int)  # 5 evenly spaced values
max_depth_range = np.linspace(1, 50, 5, dtype=int)  # 5 evenly spaced values
 
best_accuracy = 0
best_params = {}

for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)

            # Train the model
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
            model.fit(X_train, y_train)

            #Evalute the model
            accuracy = model.score(X_test, y_test)
            mlflow.log_metric("accuracy", accuracy)

            # Save the best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth
                }
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    input_example=input_example
                )