from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def train_models(X_train, y_train):
    """Entrena modelos con diferentes configuraciones de hiperparámetros.
    
    Args:
        X_train: Conjunto de entrenamiento.
        y_train: Etiquetas de entrenamiento.
    
    Returns:
        dict: Modelos entrenados con sus mejores configuraciones.
    """
    models = {
        "LogisticRegression": LogisticRegression(),
        "RandomForest": RandomForestClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
        "SVM": SVC(probability=True),
        "KNeighbors": KNeighborsClassifier()
    }
    
    params = {
        "LogisticRegression": {'C': [0.1, 1, 10]},
        "RandomForest": {'n_estimators': [50, 100], 'max_depth': [5, 10]},
        "GradientBoosting": {'learning_rate': [0.01, 0.1], 'n_estimators': [50, 100]},
        "SVM": {'C': [0.1, 1], 'kernel': ['linear', 'rbf']},
        "KNeighbors": {'n_neighbors': [3, 5, 7]}
    }
    
    best_models = {}
    for name, model in models.items():
        clf = GridSearchCV(model, params[name], cv=5)
        clf.fit(X_train, y_train)
        best_models[name] = clf.best_estimator_
    
    return best_models
