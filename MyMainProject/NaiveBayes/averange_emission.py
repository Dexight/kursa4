#Файл с заменой выбросов на среднее значение (с учётом выбросов)
from analysis import * 

print("\n ПЕРВОНАЧАЛЬНЫЕ ДАННЫЕ МЕТОДОВ КЛАССИФИКАЦИИ: ")

from NaiveBayes import NaiveBayes

print("\n" * 2)

#2) Обработка выбросов
print("ОБРАБОТКА МЕТОДОМ ЗАМЕНЫ ВЫБРОСОВ НА СРЕДНЕЕ ЗНАЧЕНИЕ:")
print("\n")

means = standardized_features.mean(0)
mean_value_6 = means[5]
mean_value_8 = means[7]
mean_value_9 = means[8]

print("Средние стандартизированные значения для K, Ba и Fe соответственно равны ", mean_value_6, " ", mean_value_8, " и ", mean_value_9)
print("\n")

#ЗАМЕНА ВЫБРОСОВ НА СРЕДНЕЕ ЗНАЧЕНИЕ

#print("БЫЛО:")
#for i in outliers_in_6[0]:
#    print(standardized_features[i][5])

for i in outliers_in_6[0]:
    standardized_features[i][5] = mean_value_6

#print("СТАЛО:")
#for i in outliers_in_6[0]:
#    print(standardized_features[i][5])

for i in outliers_in_8[0]:
    standardized_features[i][7] = mean_value_8

for i in outliers_in_9[0]:
    standardized_features[i][8] = mean_value_9

print()
print("РЕЗУЛЬТАТ:\n\n")

NaiveBayes(standardized_features, df['Class'])

# Вывод: Обработка выбросов заменой на среднее значение по признакам дала противоречивый результат.
# Данные стали менее предсказуемыми для метода логистической регрессии.

# Результат не поменялся для метода случайного леса с настройкой гиперпараметров,
# но в случае отсутствия настройки гиперпараметров ухудшился на ~2.3%.

# Стал выдавать гораздо лучший результат при использовании наивного Байеса (97.6%), что больше на 0.2% в сравнении с удалением выбросов.