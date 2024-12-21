import os
import pandas as pd
import joblib
from datetime import datetime
from configparser import ConfigParser
import mlflow

# Definir rutas y configuraciones
RUTA_DATOS_PRUEBA = r"C:\Users\jcboc\Desktop\Python 2024\TMIR\Proyecto_2_ML\data\raw\HeartDiseaseTrain-Test.csv" # Ruta de los datos
CARPETA_ARTEFACTOS = "artefactos"
CARPETA_PREDICCIONES = "data/predictions"
NOMBRE_PIPELINE_ENTRENADO = "pipeline_entrenado.pkl"
CONFIG_PATH = "config/config.cfg"

# Asegurarse de que las carpetas necesarias existan
os.makedirs(CARPETA_PREDICCIONES, exist_ok=True)

def cargar_datos(ruta_datos):
    try:
        datos = pd.read_csv(ruta_datos)
        return datos
    except FileNotFoundError:
        raise Exception(f"El archivo en {ruta_datos} no fue encontrado.")

def cargar_pipeline(ruta_pipeline):
    try:
        pipeline = joblib.load(ruta_pipeline)
        return pipeline
    except FileNotFoundError:
        raise Exception(f"El archivo en {ruta_pipeline} no fue encontrado.")

def realizar_predicciones(pipeline, datos_prueba):
    predicciones = pipeline.predict(datos_prueba)
    return predicciones

if __name__ == "__main__":
    # Cargar los datos de prueba
    datos_prueba = cargar_datos(RUTA_DATOS_PRUEBA)

    # Cargar el pipeline entrenado
    ruta_pipeline_entrenado = os.path.join(CARPETA_ARTEFACTOS, NOMBRE_PIPELINE_ENTRENADO)
    pipeline_entrenado = cargar_pipeline(ruta_pipeline_entrenado)

    # Realizar predicciones
    predicciones = realizar_predicciones(pipeline_entrenado, datos_prueba)

    # Crear un DataFrame con las predicciones
    resultados = pd.DataFrame({"predicciones": predicciones})

    # Generar el nombre del archivo con la fecha y hora actuales
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    nombre_archivo_predicciones = f"predicciones_{timestamp}.csv"
    ruta_archivo_predicciones = os.path.join(CARPETA_PREDICCIONES, nombre_archivo_predicciones)

    # Guardar las predicciones en un archivo CSV
    resultados.to_csv(ruta_archivo_predicciones, index=False)
    print(f"Predicciones guardadas en {ruta_archivo_predicciones}")

    # Registrar las predicciones en MLflow
    mlflow.set_experiment("Predicciones")  # Nombre del experimento
    with mlflow.start_run(run_name="Predicciones_HeartDisease"):  # Nombre personalizado del run
        mlflow.log_artifact(ruta_archivo_predicciones, artifact_path="predicciones")
        print(f"Predicciones registradas en MLflow.")