from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from analysis import *

def RandForest(features, Class): 
    print("\n" * 2)
    print("СЛУЧАЙНЫЙ ЛЕС")

    #=============================\
    #БЕЗ НАСТРОЙКИ ГИПЕРПАРАМЕТРОВ|
    #----------------------------/

    print("\nБез настройки гиперпараметров: ")

    # Разделение данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(features, Class, test_size=0.2, random_state=42)

    # Создание модели случайного леса
    classifier = RandomForestClassifier(random_state=42)

    # Обучение модели на обучающем наборе
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
        'n_estimators': [10, 50, 100],  # количество деревьев в лесу
        'max_depth': [None, 10, 20],      # максимальная глубина деревьев
        'min_samples_split': [2, 5, 10],  # минимальное количество образцов для разделения узла
        'min_samples_leaf': [1, 2, 4],    # минимальное количество образцов, необходимых для листового узла
    }

    # Создание модели случайного леса
    classifier = RandomForestClassifier(random_state=42)

    # Создание объекта GridSearchCV
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
    best_predictions = best_classifier.predict(X_test)

    # Оценка точности на тестовом наборе
    accuracy = accuracy_score(y_test, best_predictions)
    print("Точность модели на тестовом наборе данных с лучшими параметрами:", accuracy)

RandForest(standardized_features, df['Type'])

#Без настройки гиперпараметров:
#Точность модели на тестовом наборе данных: 0.7674418604651163

#С настройкой гиперпараметров:
#Лучшие параметры: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
#Лучшая точность на обучающем наборе: 0.7705882352941177
#Точность модели на тестовом наборе данных с лучшими параметрами: 0.7674418604651163