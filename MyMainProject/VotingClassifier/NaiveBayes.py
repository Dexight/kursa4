from sklearn.naive_bayes import GaussianNB
from analysis import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def NaiveBayes(features, Class, rs):
    print("\n" * 2)
    print("НАИВНЫЙ БАЙЕС")

    #=============================\
    #БЕЗ НАСТРОЙКИ ГИПЕРПАРАМЕТРОВ|
    #----------------------------/

    # Разделение данных на обучающий и тестовый наборы
    X_train, X_test, Y_train, Y_test = train_test_split(features, Class, test_size=0.2, random_state=rs)

    # Создание и обучение модели наивного Байеса на обучающем наборе
    classifier = GaussianNB()
    classifier.fit(X_train, Y_train)

    # Предсказание классов для тестового набора
    predictions_test = classifier.predict(X_test)

    # Оценка производительности модели на тестовом наборе
    accuracy = accuracy_score(Y_test, predictions_test)
    print("Оценка на тестовом наборе:", accuracy)
    return classifier