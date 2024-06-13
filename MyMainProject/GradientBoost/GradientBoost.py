from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from analysis import *

def GradientBoost(features, Class): 
    print("\n" * 2)
    print("ГРАДИЕНТНЫЙ БУСТИНГ")

    #=============================\
    #БЕЗ НАСТРОЙКИ ГИПЕРПАРАМЕТРОВ|
    #----------------------------/

    print("\nБез настройки гиперпараметров: ")

    # Разделение данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(features, Class, test_size=0.2, random_state=42)

    # Создание модели градиентного бустинга
    classifier = GradientBoostingClassifier(random_state=42)

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
        'n_estimators': [50, 100, 150],        # количество деревьев
        'learning_rate': [0.01, 0.1, 0.5],      # коэффициент обучения
        'max_depth': [3, 5, 7],                 # максимальная глубина деревьев
        'min_samples_split': [2, 5, 10],        # минимальное количество образцов для разделения узла
        'min_samples_leaf': [1, 2, 4]           # минимальное количество образцов в листовом узле
    }

    # Создание модели градиентного бустинга
    classifier = GradientBoostingClassifier(random_state=42)

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
    classification_report(y_test, best_predictions)

GradientBoost(standardized_features, df['Class'])

# Вывод: Получили очень высокий уровень точности.
# Модель с настройкой гиперпараметров имеет большую точность на тестовом наборе данных (~96.47%)
# по сравнению с моделью без настройки гиперпараметров (~95.35%), что является на данный момент лучшим результатом
#(при реализованных методах ЛогРег, НаивБай, СлучЛес) 

# Также имеет самую высокую точность на обучающем наборе данных (~97.67%) по сравнению с другими методами.

# В целом, данный метод работает лучше всех на моем наборе данных по сравнению со всеми другими методами.
# Возможно, при разных random_state (т.е. в разных случаях) на данном датасете ему может составить конкуренцию метод случайного леса,
# но всё же он не так сильно склонен к переобучению, нежели граниентный бустинг. Это может означать то, что
# более сложная модель на основе градиентного бустинга, при возможном изначально худшем результате чем у метода случ. леса,
# рано или поздно переобучаясь выдаст лучший результат чем случ. лес.

# Стоит также отметить долгую работу алгоритма при подборе гипперпараметров даже в сравнении с случ. лесом,
# но оно и понятно, учитывая, что модель чаще переобучается, имеет больше настраиваемых параметров и 
# реализована на использовании нескольких деревьев решений и функции минимизации потерь (|предсказанные - реальные данные| < eps).