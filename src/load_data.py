import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """Carga los datos desde un archivo CSV.
    
    Args:
        file_path (str): Ruta al archivo CSV.
    
    Returns:
        pd.DataFrame: Dataset cargado.
    """
    return pd.read_csv(file_path)
