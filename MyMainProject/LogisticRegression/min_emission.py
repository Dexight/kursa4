#Файл с заменой выбросов на минимум
from analysis import * 

print("\n ПЕРВОНАЧАЛЬНЫЕ ДАННЫЕ МЕТОДОВ КЛАССИФИКАЦИИ: ")

from LogisticRegression import LogRegr

print("\n" * 2)

#5) Обработка выбросов
print("ОБРАБОТКА МЕТОДОМ ЗАМЕНЫ ВЫБРОСОВ НА МИНИМАЛЬНОЕ ЗНАЧЕНИЕ СРЕДИ НЕ-ВЫБРОСОВ:")
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

print("Минимумы для K, Ba и Fe соответственно равны: ", min_6, " ", min_8, " ", min_9)

#ЗАМЕНА ВЫБРОСОВ НА МИНИМУМ

#print("БЫЛО:")
#for i in outliers_in_6[0]:
#    print(standardized_features[i][5])

for i in outliers_in_6[0]:
    standardized_features[i][5] = min_6

#print("СТАЛО:")
#for i in outliers_in_6[0]:
#    print(standardized_features[i][5])

for i in outliers_in_8[0]:
    standardized_features[i][7] = min_8

for i in outliers_in_9[0]:
    standardized_features[i][8] = min_9

print()
print("РЕЗУЛЬТАТ:\n\n")

LogRegr(standardized_features, df['Class'])

# Вывод: Всё работает очень плохо по сравнению со всеми другими методами.