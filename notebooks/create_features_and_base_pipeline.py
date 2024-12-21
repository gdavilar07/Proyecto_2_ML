"""
Módulo para crear y guardar un pipeline base para preprocesamiento de datos.
Este script incluye la carga de datos, división de los datos en conjuntos de entrenamiento y prueba,
y la configuración de un pipeline base para estandarizar y codificar los datos.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib

# Definir las rutas para los datos de entrada y almacenamiento de artefactos
RUTA_DATOS = (
    r"C:\Users\jcboc\Desktop\Python 2024\TMIR\Proyecto_2_ML"
    r"\data\raw\HeartDiseaseTrain-Test.csv"
)  # Ruta de los datos
CARPETA_ARTEFACTOS = "artefactos"
NOMBRE_PIPELINE = "pipeline_base.pkl"

# Asegurarse de que la carpeta de artefactos exista
os.makedirs(CARPETA_ARTEFACTOS, exist_ok=True)

def cargar_datos(ruta_datos):
    """
    Cargar el conjunto de datos desde la ruta especificada.

    Args:
        ruta_datos (str): Ruta al archivo del conjunto de datos.

    Returns:
        pd.DataFrame: Conjunto de datos cargado.
    """
    try:
        return pd.read_csv(ruta_datos)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"El archivo en {ruta_datos} no fue encontrado.") from exc

def dividir_datos(datos_local, columna_objetivo_local, proporcion_prueba=0.2, semilla=42):
    """
    Dividir el conjunto de datos en entrenamiento y prueba.

    Args:
        datos_local (pd.DataFrame): El conjunto de datos a dividir.
        columna_objetivo_local (str): El nombre de la columna objetivo.
        proporcion_prueba (float): La proporción del conjunto de prueba.
        semilla (int): Semilla para reproducibilidad.

    Returns:
        tuple: x_train, x_test, y_train, y_test
    """
    x = datos_local.drop(columns=[columna_objetivo_local])
    y = datos_local[columna_objetivo_local]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=proporcion_prueba, random_state=semilla
    )
    return x_train, x_test, y_train, y_test

def crear_pipeline_base():
    """
    Crear un pipeline base para preprocesamiento.

    Returns:
        sklearn.pipeline.Pipeline: Pipeline configurado.
    """
    columnas_categoricas = [
        "sex",
        "chest_pain_type",
        "fasting_blood_sugar",
        "rest_ecg",
        "exercise_induced_angina",
        "slope",
        "vessels_colored_by_flourosopy",
        "thalassemia",
    ]
    columnas_numericas = [
        "age",
        "resting_blood_pressure",
        "cholestoral",
        "Max_heart_rate",
        "oldpeak",
    ]

    preprocesador = ColumnTransformer(
        [
            ("cat", OneHotEncoder(), columnas_categoricas),
            ("num", StandardScaler(), columnas_numericas),
        ]
    )

    pipeline = Pipeline([
        ("preprocesador", preprocesador),
    ])
    return pipeline

def guardar_pipeline(pipeline_local, ruta_archivo):
    """
    Guardar el pipeline en un archivo.

    Args:
        pipeline_local (Pipeline): El pipeline a guardar.
        ruta_archivo (str): Ruta donde se guardará el pipeline.
    """
    joblib.dump(pipeline_local, ruta_archivo)

if __name__ == "__main__":
    # Cargar el conjunto de datos
    datos = cargar_datos(RUTA_DATOS)

    # Dividir los datos
    COLUMNA_OBJETIVO = "target"  # Nombre correcto de la columna objetivo
    x_train, x_test, y_train, y_test = dividir_datos(datos, COLUMNA_OBJETIVO)

    # Crear el pipeline base
    pipeline_base = crear_pipeline_base()

    # Guardar el pipeline en la carpeta de artefactos
    ruta_pipeline = os.path.join(CARPETA_ARTEFACTOS, NOMBRE_PIPELINE)
    guardar_pipeline(pipeline_base, ruta_pipeline)
    print(f"Pipeline base guardado en {ruta_pipeline}")
