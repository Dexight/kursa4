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
classes = df['Type']

cb, cb_gs = CatBoost(standardized_features, classes, rs)
#gb, gb_gs = GradientBoost(standardized_features, classes, rs)
lr, lr_gs = LogRegr(standardized_features, classes, rs)
nb = NaiveBayes(standardized_features, classes, rs)
#rf, rf_gs = RandForest(standardized_features, classes, rs)
#ab, ab_gs = AdaBoost(standardized_features, classes, rs)

vc = VotingClassifier(voting='hard', estimators=[('lr', lr),
                                                 ('lr_gs', lr_gs),
                                                 ('nb', nb),
                                                 #('rf', rf),
                                                 #('rf_gs', rf_gs),
                                                 ('cb', cb),
                                                 ('cb_gs', cb_gs),
                                                 #('gb', gb),
                                                 #('gb_gs', gb_gs),
                                                 #('ab', ab),
                                                 #('ab_gs', ab_gs)
                                                 ])
#т. к. добавляю несколько моделей - те, что не улучшались через GSCV, и те, что улучшались - добавлять допольнительно подбор параметров не нужно

X_train, X_test, y_train, y_test = train_test_split(standardized_features, classes, test_size=0.2, random_state=rs)
print("standardized_features.shape: \n", standardized_features.shape)
print("X_train.shape: \n", X_train.shape)
print("X_test.shape \n", X_test.shape)
vc.fit(X_train, y_train)
predictions = vc.predict(X_test)

# Оценка точности модели
accuracy = accuracy_score(y_test, predictions)
print("Точность модели на тестовом наборе данных:", accuracy)

crstab = pd.crosstab(y_test, predictions)

# Traceback (most recent call last):
#  File "c:\MyProject2\VotingClassifier\VotingClassifier.py", line 37, in <module>
#    predictions = vc.predict(X_test)
#                  ^^^^^^^^^^^^^^^^^^
#  File "c:\Users\mrdim\AppData\Local\Programs\Python\Python312\Lib\site-packages\sklearn\ensemble\_voting.py", line 369, in predict
#    predictions = self._predict(X)
#                  ^^^^^^^^^^^^^^^^
#  File "c:\Users\mrdim\AppData\Local\Programs\Python\Python312\Lib\site-packages\sklearn\ensemble\_voting.py", line 68, in _predict
#    return np.asarray([est.predict(X) for est in self.estimators_]).T
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 2 dimensions. The detected shape was (11, 43) + inhomogeneous part.