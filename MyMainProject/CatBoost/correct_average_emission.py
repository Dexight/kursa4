#Файл с заменой выбросов на среднее значение (без учёта выбросов)
from analysis import *

print("\n ПЕРВОНАЧАЛЬНЫЕ ДАННЫЕ МЕТОДОВ КЛАССИФИКАЦИИ: ")

from CatBoost import CatBoost

print("\n" * 2)

#4) Обработка выбросов
print("ОБРАБОТКА МЕТОДОМ ЗАМЕНЫ ВЫБРОСОВ НА СРЕДНЕЕ ЗНАЧЕНИЕ:")
print("\n")

sum_6 = np.sum(standardized_features[:, 5])
sum_8 = np.sum(standardized_features[:, 7])
sum_9 = np.sum(standardized_features[:, 8])

for i in outliers_in_6[0]:
    sum_6 -= standardized_features[i][5]
for i in outliers_in_8[0]:
    sum_6 -= standardized_features[i][7]
for i in outliers_in_9[0]:
    sum_6 -= standardized_features[i][8]

count_6 = standardized_features[:, 5].size - outliers_in_6[0].size
count_8 = standardized_features[:, 7].size - outliers_in_8[0].size
count_9 = standardized_features[:, 8].size - outliers_in_9[0].size

mean_value_6 = sum_6/count_6
mean_value_8 = sum_8/count_8
mean_value_9 = sum_9/count_9

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

CatBoost(standardized_features, df['Class'])