import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from configparser import ConfigParser

# Definir rutas para datos y artefactos
RUTA_DATOS = r"C:\Users\jcboc\Desktop\Python 2024\TMIR\Proyecto_2_ML\data\raw\HeartDiseaseTrain-Test.csv" # Ruta de los datos
CARPETA_ARTEFACTOS = "artefactos"
NOMBRE_PIPELINE = "pipeline_entrenado.pkl"
CONFIG_PATH = "config/config.cfg"

# Asegurarse de que la carpeta de artefactos exista
os.makedirs(CARPETA_ARTEFACTOS, exist_ok=True)

def cargar_datos(ruta_datos):
    """
    Cargar el conjunto de datos desde la ruta especificada.

    Parámetros:
        ruta_datos (str): Ruta al archivo del conjunto de datos.

    Retorna:
        pd.DataFrame: Conjunto de datos cargado.
    """
    try:
        datos = pd.read_csv(ruta_datos)
        return datos
    except FileNotFoundError:
        raise Exception(f"El archivo en {ruta_datos} no fue encontrado.")

def cargar_pipeline(ruta_pipeline):
    """
    Cargar un pipeline guardado en un archivo .pkl.

    Parámetros:
        ruta_pipeline (str): Ruta al archivo del pipeline.

    Retorna:
        Pipeline: Pipeline cargado.
    """
    try:
        pipeline = joblib.load(ruta_pipeline)
        return pipeline
    except FileNotFoundError:
        raise Exception(f"El archivo en {ruta_pipeline} no fue encontrado.")

def entrenar_pipeline(pipeline, X_train, y_train, config):
    """
    Entrenar el pipeline con el modelo configurado.

    Parámetros:
        pipeline (Pipeline): Pipeline base cargado.
        X_train (pd.DataFrame): Datos de entrenamiento.
        y_train (pd.Series): Etiquetas de entrenamiento.
        config (ConfigParser): Configuración del modelo.

    Retorna:
        Pipeline: Pipeline entrenado.
    """
    modelo = RandomForestClassifier(
        n_estimators=config.getint('RandomForest', 'n_estimators', fallback=100),
        max_depth=config.getint('RandomForest', 'max_depth', fallback=None),
        random_state=config.getint('General', 'random_state', fallback=42)
    )
    pipeline.steps.append(('modelo', modelo))
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluar_modelo(pipeline, X_test, y_test):
    """
    Evaluar el pipeline entrenado.

    Parámetros:
        pipeline (Pipeline): Pipeline entrenado.
        X_test (pd.DataFrame): Datos de prueba.
        y_test (pd.Series): Etiquetas de prueba.

    Retorna:
        dict: Métricas de evaluación (accuracy y F1-score).
    """
    predicciones = pipeline.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, predicciones),
        "f1_score": f1_score(y_test, predicciones)
    }

if __name__ == "__main__":
    # Cargar el conjunto de datos
    datos = cargar_datos(RUTA_DATOS)

    # Dividir los datos
    columna_objetivo = "target"  # Nombre correcto de la columna objetivo
    X = datos.drop(columns=[columna_objetivo])
    y = datos[columna_objetivo]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Cargar el pipeline base
    ruta_pipeline_base = os.path.join(CARPETA_ARTEFACTOS, "pipeline_base.pkl")
    pipeline_base = cargar_pipeline(ruta_pipeline_base)

    # Cargar la configuración
    config = ConfigParser()
    config.read(CONFIG_PATH)

    # Entrenar el pipeline
    pipeline_entrenado = entrenar_pipeline(pipeline_base, X_train, y_train, config)

    # Guardar el pipeline entrenado
    ruta_pipeline_entrenado = os.path.join(CARPETA_ARTEFACTOS, NOMBRE_PIPELINE)
    joblib.dump(pipeline_entrenado, ruta_pipeline_entrenado)
    print(f"Pipeline entrenado guardado en {ruta_pipeline_entrenado}")

    # Evaluar el modelo
    metricas = evaluar_modelo(pipeline_entrenado, X_test, y_test)
    print("Métricas de evaluación:", metricas)