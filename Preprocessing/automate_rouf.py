import numpy as np
import pandas as pd
import mlflow

def preprocess_data(df):
    """
    Preprocess the diabetes dataset by replacing zero values in specific columns with NaN.
    Args:
        df (pd.DataFrame): The input DataFrame containing the diabetes dataset.
    Returns:
        pd.DataFrame: The preprocessed DataFrame with zero values replaced by NaN in specified columns.
    """
    feature = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    df[feature] = df[feature].replace(0, np.nan)

    # Fill NaN values with the mean of each column
    df[feature] = df[feature].fillna(df[feature].median())
    
    return df


# Mencoba fungsi preprocess_data
if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv("diabetes.csv")
    
    # Preprocess the data
    preprocessed_df = preprocess_data(df)
    
    # Save the preprocessed data to a new CSV file
    preprocessed_df.to_csv("preprocessed_diabetes.csv", index=False)
    
    print("Preprocessing complete. The preprocessed data has been saved to 'preprocessed_diabetes.csv'.")

