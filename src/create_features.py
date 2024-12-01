from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd

def preprocess_data(data: pd.DataFrame, target: str):
    """Preprocesa los datos aplicando imputación, codificación y estandarización.

    Args:
        data (pd.DataFrame): Dataset original.
        target (str): Columna objetivo.
    
    Returns:
        tuple: Datos preprocesados (X_train, X_test, y_train, y_test, preprocessor).
    """
    X = data.drop(columns=[target])
    y = data[target]
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Definir columnas categóricas y numéricas
    categorical_features = ['chest_pain_type', 'fasting_blood_sugar', 'rest_ecg']
    numerical_features = ['age', 'resting_blood_pressure', 'cholestoral', 'oldpeak']
    
    # Transformaciones
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Aplicar las transformaciones
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    return X_train, X_test, y_train, y_test, preprocessor
