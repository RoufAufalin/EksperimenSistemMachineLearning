import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
import warnings
import sys

# mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# mlflow.set_experiment("Diabetes Prediction Experiment")

# # Mengubah nama pengguna sesuai dengan ID Dicoding
# os.environ['MLFLOW_TRACKING_USERNAME'] = 'Rouf Aufalin'

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocessed_diabetes.csv")
    data = pd.read_csv(file_path)



    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("Outcome", axis=1),
        data["Outcome"],
        test_size=0.2,
        random_state=42
    )

    input_example = X_train[0:5]

    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 37
 

    with mlflow.start_run():
        # Train the model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)

        predicted_qualities = model.predict(X_test)

        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
            
        )

        model.fit(X_train, y_train)

        #Log metrics
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
