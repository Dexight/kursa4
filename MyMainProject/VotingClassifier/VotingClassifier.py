from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
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

print("CLASSIFICATION REPORT:")
print(classification_report(y_test, predictions))
#VOTING CLASSIFIER RESULTS:
#Точность модели на тестовом наборе данных: 0.9534883720930233
#col_0       negative   positive
#Class
# negative         34          0
# positive          2          7