from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from analysis import *

def CatBoost(features, Class):
    print("\n" * 2)
    print("МОДЕЛЬ CATBOOST")

    #=============================\
    #БЕЗ НАСТРОЙКИ ГИПЕРПАРАМЕТРОВ|
    #----------------------------/

    print("\nБез настройки гиперпараметров: ")

    # Разделение данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(features, Class, test_size=0.2, random_state=42)

    # Создание и обучение модели CatBoost
    classifier = CatBoostClassifier(random_seed=42, logging_level='Silent') # logging_level - параметр для отладки. Silent - никакого вывода, Verbose - подробный вывод
    classifier.fit(X_train, y_train)

    # Предсказание классов для тестового набора
    predictions = classifier.predict(X_test)

    # Оценка точности модели
    accuracy = accuracy_score(y_test, predictions)
    print("Точность модели на тестовом наборе данных:", accuracy)

    #============================\
    #С НАСТРОЙКОЙ ГИПЕРПАРАМЕТРОВ|
    #---------------------------/

    print("\nС настройкой гиперпараметров: ")


    # Задаем сетку гиперпараметров для перебора
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1], # Скорость обучения
        'depth': [4, 6, 8],                 # Глубина деревьев
        'iterations': [50, 100, 200]        # Количество итераций
    }

    classifier = CatBoostClassifier(random_seed=42, logging_level='Silent')

    # Инициализируем GridSearchCV
    grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy')

    # Поиск лучших параметров
    grid_search.fit(X_train, y_train)

    # Получение лучших параметров и оценки
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Лучшие параметры:", best_params)
    print("Лучшая точность на обучающем наборе:", best_score)

    # Предсказание классов для тестового набора с лучшими параметрами
    best_classifier = grid_search.best_estimator_
    predictions = best_classifier.predict(X_test)

    # Оценка точности модели на тестовом наборе
    accuracy = accuracy_score(y_test, predictions)
    print("Точность модели на тестовом наборе данных с лучшими параметрами:", accuracy)

CatBoost(standardized_features, df['Type'])
#Без настройки гиперпараметров:
#Точность модели на тестовом наборе данных: 0.7441860465116279

#С настройкой гиперпараметров:
#Лучшие параметры: {'depth': 6, 'iterations': 200, 'learning_rate': 0.1}
#Лучшая точность на обучающем наборе: 0.7588235294117648
#Точность модели на тестовом наборе данных с лучшими параметрами: 0.7674418604651163