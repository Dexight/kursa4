#Файл с заменой выбросов на максимум или минимум
from analysis import *

print("\n ПЕРВОНАЧАЛЬНЫЕ ДАННЫЕ МЕТОДОВ КЛАССИФИКАЦИИ: ")

from AdaBoost import AdaBoost

print("\n" * 2)

#7) Обработка выбросов
print("ОБРАБОТКА МЕТОДОМ ЗАМЕНЫ ВЫБРОСОВ НА МАКСИМАЛЬНОЕ и МИНИМАЛЬНОЕ ЗНАЧЕНИЕ СРЕДИ НЕ-ВЫБРОСОВ:")
print("\n")

arr_6 = standardized_features[:, 5]
arr_8 = standardized_features[:, 7]
arr_9 = standardized_features[:, 8]

# Удаление строк с выбросами
cleaned_features_6 = np.delete(arr_6, outliers_in_6, axis=0)
cleaned_features_8 = np.delete(arr_8, outliers_in_8, axis=0)
cleaned_features_9 = np.delete(arr_9, outliers_in_9, axis=0)

min_6 = cleaned_features_6.min()
min_8 = cleaned_features_8.min()
min_9 = cleaned_features_9.min()

max_6 = cleaned_features_6.max()
max_8 = cleaned_features_8.max()
max_9 = cleaned_features_9.max()

print("Минимумы для K, Ba и Fe соответственно равны: ", min_6, " ", min_8, " ", min_9)
print("Максимумы для K, Ba и Fe соответственно равны: ", max_6, " ", max_8, " ", max_9)

#ЗАМЕНА

print("БЫЛО:")
for i in outliers_in_6[0]:
    print(standardized_features[i][5])

for i in outliers_in_6[0]:
    if standardized_features[i][5] >= max_6:
        standardized_features[i][5] = max_6
    elif standardized_features[i][5] <= min_6:
        standardized_features[i][5] = min_6

print("СТАЛО:")
for i in outliers_in_6[0]:
    print(standardized_features[i][5])

for i in outliers_in_8[0]:
    if standardized_features[i][7] >= max_8:
        standardized_features[i][7] = max_8
    elif standardized_features[i][7] <= min_8:
        standardized_features[i][7] = min_8

for i in outliers_in_9[0]:
    if standardized_features[i][8] >= max_9:
        standardized_features[i][8] = max_9
    elif standardized_features[i][8] <= min_9:
        standardized_features[i][8] = min_9

print()
print("РЕЗУЛЬТАТ:\n\n")

AdaBoost(standardized_features, df['Class'])

#Вывод: Всё работает с приемлемой точностью, но в сравнении с другими методами, конечно, работает не очень.
#Результаты примерно такие же, как и при замене на медиану.