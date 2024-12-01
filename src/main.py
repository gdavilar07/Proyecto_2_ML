from load_data import load_data
from create_features import preprocess_data
from create_models import train_models

def main():
    """Pipeline principal del proyecto."""
    # Cargar los datos
    data = load_data('data/raw/HeartDiseaseTrain-Test.csv')
    
    # Preprocesar los datos
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(data, target='target')
    
    # Entrenar modelos
    best_models = train_models(X_train, y_train)
    
    # Evaluar los modelos
    for name, model in best_models.items():
        print(f"Modelo: {name}, Score: {model.score(X_test, y_test)}")

if __name__ == '__main__':
    main()
