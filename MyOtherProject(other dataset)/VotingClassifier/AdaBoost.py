from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from analysis import *

def AdaBoost(features, Class, rs): 
    print("\n" * 2)
    print("АДАПТИВНЫЙ БУСТИНГ")

    #=============================\
    #БЕЗ НАСТРОЙКИ ГИПЕРПАРАМЕТРОВ|
    #----------------------------/

    print("\nБез настройки гиперпараметров: ")

    # Разделение данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(features, Class, test_size=0.2, random_state=rs)

    # Создание модели адаптивного бустинга
    classifier = AdaBoostClassifier(random_state=rs)

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
        'learning_rate': [0.01, 0.1, 1.0],  # коэффициент обучения
    }

    # Создание модели адаптивного бустинга
    classifier = AdaBoostClassifier(random_state=rs)

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
    return [classifier, grid_search]
