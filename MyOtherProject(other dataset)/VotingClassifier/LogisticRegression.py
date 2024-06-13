from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from analysis import * 

# В этом методе очень важна стандартизация признаков (что было уже сделано заранее)
# Примечание: метод быстрее работает на большом наборе данных
def LogRegr(features, Class, rs):
    print("\n" * 2)
    print("ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ\n")
    #=============================\
    #БЕЗ НАСТРОЙКИ ГИПЕРПАРАМЕТРОВ|
    #----------------------------/
    print("Без подбора гиперпараметров:")

    # Разделение данных на обучающий и тестовый наборы
    X_train, X_test, Y_train, Y_test = train_test_split(features, Class, test_size=0.2, random_state=rs)

    # Создание и обучение модели без подбора гиперпараметров
    classifier = LogisticRegression()
    classifier.fit(X_train, Y_train)

    # Оценка производительности модели на тестовом наборе
    accuracy = classifier.score(X_test, Y_test)
    print("Рeзультат оценки на тестовом наборе данных:", accuracy)

    #===========================\
    # С ПОДБОРОМ ГИПЕРПАРАМЕТРОВ|
    #--------------------------/

    print ("\n")
    print("С подбором гиперпараметров:")

    # Создание и обучение модели с подбором гиперпараметров
    param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, Y_train)

    # Оценка производительности модели с подбором гиперпараметров методом кросс-валидации
    # (разделение доступных данных на несколько частей, последовательное использовании каждого из них в качестве тестового набора данных)
    best_accuracy = grid_search.best_score_
    print("Результат 'лучшей' оценки:", best_accuracy)

    # Оценка производительности модели на тестовом наборе с лучшими параметрами
    test_accuracy = grid_search.score(X_test, Y_test)
    print("Результат оценки на тестовом наборе данных:", test_accuracy)
    return [classifier, grid_search]