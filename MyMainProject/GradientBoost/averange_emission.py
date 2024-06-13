#Файл с заменой выбросов на среднее значение (с учётом выбросов)
from analysis import * 

print("\n ПЕРВОНАЧАЛЬНЫЕ ДАННЫЕ МЕТОДОВ КЛАССИФИКАЦИИ: ")

from GradientBoost import GradientBoost

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

GradientBoost(standardized_features, df['Class'])