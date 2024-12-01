import pandas as pd
from create_features import preprocess_data
from create_models import train_models

def predict(file_path: str, model):
    """Realiza predicciones en nuevos datos.
    
    Args:
        file_path (str): Ruta al archivo CSV con los nuevos datos.
        model: Modelo preentrenado.
    
    Returns:
        pd.DataFrame: Predicciones realizadas.
    """
    data = pd.read_csv(file_path)
    X = preprocess_data(data, target='target')[0]  # Solo transformar los datos
    predictions = model.predict(X)
    return pd.DataFrame(predictions, columns=['Predictions'])
