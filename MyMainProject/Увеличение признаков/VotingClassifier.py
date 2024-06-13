from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from analysis import *
from CatBoost import CatBoost
from GradientBoost import GradientBoost
from LogisticRegression import LogRegr
from NaiveBayes import NaiveBayes
from RandomForest import RandForest
from AdaBoost import AdaBoost
from sklearn.ensemble import VotingClassifier

rs = 42
classes = df['Class']

cb, cb_gs = CatBoost(standardized_features, classes, rs)
gb, gb_gs = GradientBoost(standardized_features, classes, rs)
lr, lr_gs = LogRegr(standardized_features, classes, rs)
nb = NaiveBayes(standardized_features, classes, rs)
rf, rf_gs = RandForest(standardized_features, classes, rs)
ab, ab_gs = AdaBoost(standardized_features, classes, rs)

vc = VotingClassifier(voting='hard', estimators=[('lr', lr),
                                                 ('lr_gs', lr_gs),
                                                 ('nb', nb),
                                                 ('rf', rf),
                                                 ('rf_gs', rf_gs),
                                                 ('cb', cb),
                                                 ('cb_gs', cb_gs),
                                                 ('gb', gb),
                                                 ('gb_gs', gb_gs),
                                                 ('ab', ab),
                                                 ('ab_gs', ab_gs)])
#т. к. добавляю несколько моделей - те, что не улучшались через GSCV, и те, что улучшались - добавлять допольнительно подбор параметров не нужно

X_train, X_test, y_train, y_test = train_test_split(standardized_features, classes, test_size=0.2, random_state=rs)
vc.fit(X_train, y_train)
predictions = vc.predict(X_test)

print("VOTING CLASSIFIER RESULTS:")

# Оценка точности модели
accuracy = accuracy_score(y_test, predictions)
print("Точность модели на тестовом наборе данных:", accuracy)

crstab = pd.crosstab(y_test, predictions)
print(crstab)

#VOTING CLASSIFIER RESULTS:
#Точность модели на тестовом наборе данных: 0.9534883720930233
#col_0       negative   positive
#Class
# negative         34          0
# positive          2          7

#Результаты для моделей с увеличением количества признаков

#CatBoost:
#Без подбора гиперпараметров
#Точность модели на тестовом наборе данных: 0.9767441860465116

#С подбором гиперпараметров
#Лучшая точность на обучающем наборе: 0.9647058823529411
#Точность модели на тестовом наборе данных с лучшими параметрами: 0.9534883720930233

#Gradient Boost:
#Без подбора гиперпараметров
#Точность модели на тестовом наборе данных: 0.9302325581395349

#С подбором гиперпараметров
#Лучшие параметры: {'learning_rate': 0.5, 'max_depth': 3, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 50}
#Лучшая точность на обучающем наборе: 0.9647058823529411
#Точность модели на тестовом наборе данных с лучшими параметрами: 0.9534883720930233

#LogReg
#Без подбора гиперпараметров:
#Рeзультат оценки на тестовом наборе данных: 0.9069767441860465

#С подбором гиперпараметров
#Результат 'лучшей' оценки: 0.9294117647058824
#Результат оценки на тестовом наборе данных: 0.9069767441860465

#NaiveBayes
#Оценка на тестовом наборе: 0.9069767441860465

#RandomForest
#Без подбора гиперпараметров:
#Точность модели на тестовом наборе данных: 0.9767441860465116

#С настройкой гиперпараметров:
#Лучшие параметры: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
#Лучшая точность на обучающем наборе: 0.9647058823529411
#Точность модели на тестовом наборе данных с лучшими параметрами: 0.9767441860465116

#AdaBoost
#Без подбора гиперпараметров:
#Точность модели на тестовом наборе данных: 0.9302325581395349

#С настройкой гиперпараметров:
#Лучшие параметры: {'learning_rate': 0.01, 'n_estimators': 10}
#Лучшая точность на обучающем наборе: 0.9647058823529411
#Точность модели на тестовом наборе данных с лучшими параметрами: 0.9767441860465116

#VOTING CLASSIFIER RESULTS:
#Точность модели на тестовом наборе данных: 0.9767441860465116
#col_0       negative   positive
#Class
# negative         34          0
# positive          1          8