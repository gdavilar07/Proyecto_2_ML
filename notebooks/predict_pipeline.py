"""
Módulo para realizar predicciones utilizando un pipeline entrenado.
Este script carga un pipeline entrenado, realiza predicciones
y registra los resultados en MLflow.
"""

import os
from datetime import datetime
import pandas as pd
import joblib
import mlflow

# Definir rutas absolutas
BASE_DIR = r"C:\Users\jcboc\Desktop\Python 2024\TMIR\Proyecto_2_ML"
RUTA_DATOS_PRUEBA = os.path.join(BASE_DIR, "data", "raw", "HeartDiseaseTrain-Test.csv")
CARPETA_ARTEFACTOS = os.path.join(BASE_DIR, "artefactos")
CARPETA_PREDICCIONES = os.path.join(BASE_DIR, "data", "predictions")
NOMBRE_PIPELINE_ENTRENADO = "pipeline_entrenado.pkl"

# Asegurarse de que las carpetas necesarias existan
os.makedirs(CARPETA_PREDICCIONES, exist_ok=True)

def cargar_datos(ruta_datos):
    """
    Carga un archivo CSV en un DataFrame.

    Args:
        ruta_datos (str): Ruta del archivo CSV.

    Returns:
        pd.DataFrame: Datos cargados en un DataFrame.
    """
    try:
        return pd.read_csv(ruta_datos)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"El archivo en {ruta_datos} no fue encontrado.") from exc

def cargar_pipeline(ruta_pipeline):
    """
    Carga un pipeline guardado en un archivo .pkl.

    Args:
        ruta_pipeline (str): Ruta del archivo del pipeline.

    Returns:
        object: Pipeline cargado.
    """
    try:
        return joblib.load(ruta_pipeline)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"El archivo en {ruta_pipeline} no fue encontrado.") from exc

def realizar_predicciones(pipeline, datos):
    """
    Realiza predicciones con un pipeline dado.

    Args:
        pipeline (object): Pipeline cargado.
        datos (pd.DataFrame): Datos para predicción.

    Returns:
        np.ndarray: Predicciones generadas.
    """
    return pipeline.predict(datos)

if __name__ == "__main__":
    # Cargar los datos de prueba
    datos_prueba = cargar_datos(RUTA_DATOS_PRUEBA)

    # Cargar el pipeline entrenado
    ruta_pipeline_local = os.path.join(CARPETA_ARTEFACTOS, NOMBRE_PIPELINE_ENTRENADO)
    pipeline_local = cargar_pipeline(ruta_pipeline_local)

    # Realizar predicciones
    predicciones = realizar_predicciones(pipeline_local, datos_prueba)

    # Crear un DataFrame con las predicciones
    resultados = pd.DataFrame({"predicciones": predicciones})

    # Generar el nombre del archivo con la fecha y hora actuales
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    nombre_archivo = f"predicciones_{timestamp}.csv"
    ruta_archivo = os.path.join(CARPETA_PREDICCIONES, nombre_archivo)

    # Guardar las predicciones en un archivo CSV
    resultados.to_csv(ruta_archivo, index=False)
    print(f"Predicciones guardadas en {ruta_archivo}")

    # Registrar las predicciones en MLflow
    mlflow.set_experiment("Predicciones")
    with mlflow.start_run(run_name="Predicciones_HeartDisease"):
        mlflow.log_artifact(ruta_archivo, artifact_path="predicciones")
        print("Predicciones registradas en MLflow.")
