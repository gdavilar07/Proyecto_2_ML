import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Define the paths for input data and artifact storage
DATA_PATH = "data/input_data.csv"  # Update with your actual data path
ARTIFACTS_DIR = "artifacts"
PIPELINE_FILENAME = "base_pipeline.pkl"

# Ensure the artifacts directory exists
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def load_data(data_path):
    """
    Load the dataset from the specified path.

    Parameters:
        data_path (str): Path to the dataset file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        data = pd.read_csv(data_path)
        return data
    except FileNotFoundError:
        raise Exception(f"The file at {data_path} was not found.")

def split_data(data, target_column, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    Parameters:
        data (pd.DataFrame): The dataset to split.
        target_column (str): The name of the target column.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def create_base_pipeline():
    """
    Create a base pipeline for preprocessing.

    Returns:
        sklearn.pipeline.Pipeline: Configured pipeline.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
    ])
    return pipeline

def save_pipeline(pipeline, filename):
    """
    Save the pipeline to a file.

    Parameters:
        pipeline (Pipeline): The pipeline to save.
        filename (str): Path where the pipeline will be saved.
    """
    joblib.dump(pipeline, filename)

if __name__ == "__main__":
    # Load the dataset
    data = load_data(DATA_PATH)

    # Split the data
    target_column = "target"  # Replace with your target column name
    X_train, X_test, y_train, y_test = split_data(data, target_column)

    # Create the base pipeline
    base_pipeline = create_base_pipeline()

    # Save the pipeline to the artifacts directory
    pipeline_path = os.path.join(ARTIFACTS_DIR, PIPELINE_FILENAME)
    save_pipeline(base_pipeline, pipeline_path)
    print(f"Base pipeline saved to {pipeline_path}")