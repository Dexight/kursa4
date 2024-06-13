#Файл с заменой выбросов на максимум
from analysis import *

print("\n ПЕРВОНАЧАЛЬНЫЕ ДАННЫЕ МЕТОДОВ КЛАССИФИКАЦИИ: ")

from AdaBoost import AdaBoost

print("\n" * 2)

#6) Обработка выбросов
print("ОБРАБОТКА МЕТОДОМ ЗАМЕНЫ ВЫБРОСОВ НА МАКСИМАЛЬНОЕ ЗНАЧЕНИЕ СРЕДИ НЕ-ВЫБРОСОВ:")
print("\n")

arr_6 = standardized_features[:, 5]
arr_8 = standardized_features[:, 7]
arr_9 = standardized_features[:, 8]

# Удаление строк с выбросами
cleaned_features_6 = np.delete(arr_6, outliers_in_6, axis=0)
cleaned_features_8 = np.delete(arr_8, outliers_in_8, axis=0)
cleaned_features_9 = np.delete(arr_9, outliers_in_9, axis=0)

max_6 = cleaned_features_6.max()
max_8 = cleaned_features_8.max()
max_9 = cleaned_features_9.max()

print("Максимумы для K, Ba и Fe соответственно равны: ", max_6, " ", max_8, " ", max_9)

#ЗАМЕНА ВЫБРОСОВ НА МАКСИМУМ

#print("БЫЛО:")
#for i in outliers_in_6[0]:
#    print(standardized_features[i][5])

for i in outliers_in_6[0]:
    standardized_features[i][5] = max_6

#print("СТАЛО:")
#for i in outliers_in_6[0]:
#    print(standardized_features[i][5])

for i in outliers_in_8[0]:
    standardized_features[i][7] = max_8

for i in outliers_in_9[0]:
    standardized_features[i][8] = max_9

print()
print("РЕЗУЛЬТАТ:\n\n")

AdaBoost(standardized_features, df['Class'])

# Вывод: замена на максимальное значение не дало особо высоких результатов.
# Всё так же хорошо работает случайный лес, лучше стал работать Наивный Байес, но логистическая регрессия работает хуже.