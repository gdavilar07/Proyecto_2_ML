import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Definir las rutas para los datos de entrada y almacenamiento de artefactos
RUTA_DATOS = "../data/raw/HeartDiseaseTrain-Test.csv"  # Actualizar con la ruta real de los datos
CARPETA_ARTEFACTOS = "artefactos"
NOMBRE_PIPELINE = "pipeline_base.pkl"

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

def dividir_datos(datos, columna_objetivo, proporcion_prueba=0.2, semilla=42):
    """
    Dividir el conjunto de datos en entrenamiento y prueba.

    Parámetros:
        datos (pd.DataFrame): El conjunto de datos a dividir.
        columna_objetivo (str): El nombre de la columna objetivo.
        proporcion_prueba (float): La proporción del conjunto de prueba.
        semilla (int): Semilla para reproducibilidad.

    Retorna:
        tuple: X_entrenamiento, X_prueba, y_entrenamiento, y_prueba
    """
    X = datos.drop(columns=[columna_objetivo])
    y = datos[columna_objetivo]
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=proporcion_prueba, random_state=semilla)
    return X_entrenamiento, X_prueba, y_entrenamiento, y_prueba

def crear_pipeline_base():
    """
    Crear un pipeline base para preprocesamiento.

    Retorna:
        sklearn.pipeline.Pipeline: Pipeline configurado.
    """
    pipeline = Pipeline([
        ('escalador', StandardScaler()),
    ])
    return pipeline

def guardar_pipeline(pipeline, nombre_archivo):
    """
    Guardar el pipeline en un archivo.

    Parámetros:
        pipeline (Pipeline): El pipeline a guardar.
        nombre_archivo (str): Ruta donde se guardará el pipeline.
    """
    joblib.dump(pipeline, nombre_archivo)

if __name__ == "__main__":
    # Cargar el conjunto de datos
    datos = cargar_datos(RUTA_DATOS)

    # Dividir los datos
    columna_objetivo = "HeartDisease"  # Reemplazar con el nombre real de la columna objetivo
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = dividir_datos(datos, columna_objetivo)

    # Crear el pipeline base
    pipeline_base = crear_pipeline_base()

    # Guardar el pipeline en la carpeta de artefactos
    ruta_pipeline = os.path.join(CARPETA_ARTEFACTOS, NOMBRE_PIPELINE)
    guardar_pipeline(pipeline_base, ruta_pipeline)
    print(f"Pipeline base guardado en {ruta_pipeline}")