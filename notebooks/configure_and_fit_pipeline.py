"""
Módulo para configurar y entrenar un pipeline utilizando datos de entrenamiento.
Este script carga un pipeline base, ajusta un modelo y lo guarda junto con las métricas
obtenidas en los datos de prueba.
"""

import os
from configparser import ConfigParser
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Definir rutas para datos y artefactos
RUTA_DATOS = (
    r"C:\Users\jcboc\Desktop\Python 2024\TMIR\Proyecto_2_ML"
    r"\data\raw\HeartDiseaseTrain-Test.csv"
)  # Ruta de los datos
CARPETA_ARTEFACTOS = "artefactos"
NOMBRE_PIPELINE = "pipeline_entrenado.pkl"
CONFIG_PATH = "config/config.cfg"

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


def cargar_pipeline(ruta_pipeline):
    """
    Cargar un pipeline guardado en un archivo .pkl.

    Args:
        ruta_pipeline (str): Ruta al archivo del pipeline.

    Returns:
        object: Pipeline cargado.
    """
    try:
        return joblib.load(ruta_pipeline)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"El archivo en {ruta_pipeline} no fue encontrado.") from exc


def entrenar_pipeline(pipeline, x_train_local, y_train_local, config_local):
    """
    Entrenar el pipeline con el modelo configurado.

    Args:
        pipeline (Pipeline): Pipeline base cargado.
        x_train_local (pd.DataFrame): Datos de entrenamiento.
        y_train_local (pd.Series): Etiquetas de entrenamiento.
        config_local (ConfigParser): Configuración del modelo.

    Returns:
        Pipeline: Pipeline entrenado.
    """
    modelo = RandomForestClassifier(
        n_estimators=config_local.getint("RandomForest", "n_estimators", fallback=100),
        max_depth=config_local.getint("RandomForest", "max_depth", fallback=None),
        random_state=config_local.getint("General", "random_state", fallback=42),
    )
    pipeline.steps.append(("modelo", modelo))
    pipeline.fit(x_train_local, y_train_local)
    return pipeline


def evaluar_modelo(pipeline, x_test_local, y_test_local):
    """
    Evaluar el pipeline entrenado.

    Args:
        pipeline (Pipeline): Pipeline entrenado.
        x_test_local (pd.DataFrame): Datos de prueba.
        y_test_local (pd.Series): Etiquetas de prueba.

    Returns:
        dict: Métricas de evaluación (accuracy y F1-score).
    """
    predicciones = pipeline.predict(x_test_local)
    return {
        "accuracy": accuracy_score(y_test_local, predicciones),
        "f1_score": f1_score(y_test_local, predicciones),
    }


if __name__ == "__main__":
    # Cargar el conjunto de datos
    datos = cargar_datos(RUTA_DATOS)

    # Dividir los datos
    COLUMNA_OBJETIVO = "target"  # Nombre correcto de la columna objetivo
    x = datos.drop(columns=[COLUMNA_OBJETIVO])
    y = datos[COLUMNA_OBJETIVO]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Cargar el pipeline base
    ruta_pipeline_base = os.path.join(CARPETA_ARTEFACTOS, "pipeline_base.pkl")
    pipeline_base = cargar_pipeline(ruta_pipeline_base)

    # Cargar la configuración
    config = ConfigParser()
    config.read(CONFIG_PATH)

    # Entrenar el pipeline
    pipeline_entrenado = entrenar_pipeline(pipeline_base, x_train, y_train, config)

    # Guardar el pipeline entrenado
    ruta_pipeline_entrenado = os.path.join(CARPETA_ARTEFACTOS, NOMBRE_PIPELINE)
    joblib.dump(pipeline_entrenado, ruta_pipeline_entrenado)
    print(f"Pipeline entrenado guardado en {ruta_pipeline_entrenado}")

    # Evaluar el modelo
    metricas = evaluar_modelo(pipeline_entrenado, x_test, y_test)
    print("Métricas de evaluación:", metricas)
