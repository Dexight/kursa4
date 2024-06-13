#Файл с обработкой выбросов
from analysis import * 

print("\n ПЕРВОНАЧАЛЬНЫЕ ДАННЫЕ МЕТОДОВ КЛАССИФИКАЦИИ: ")

from RandomForest import RandForest

print("\n" * 2)

#8) Обработка выбросов
print("ОБРАБОТКА МЕТОДОМ ЗАМЕНЫ ВЫБРОСОВ НА МЕДИАНУ:")
print("\n")

median_value_6 = np.median(standardized_features[:][5])
median_value_8 = np.median(standardized_features[:][7])
median_value_9 = np.median(standardized_features[:][8])

print("Медианные стандартизированные значения для K, Ba и Fe соответственно равны ", median_value_6, " ", median_value_8, " и ", median_value_9)
print("\n")

#ЗАМЕНА ВЫБРОСОВ НА МЕДИАНУ

for i in outliers_in_6[0]:
    standardized_features[i][5] = median_value_6

for i in outliers_in_8[0]:
    standardized_features[i][7] = median_value_8

for i in outliers_in_9[0]:
    standardized_features[i][8] = median_value_9

print()
print("РЕЗУЛЬТАТ:\n\n")

RandForest(standardized_features, df['Class'])

# Вывод: При замене на медианные значения стал хуже работать метод логистической регрессии,
# но результат всё равно достаточно высокий по сравнению с методом замены на среднее значение.

# Наивный Байес стал работать лучше, но разница составляет всего ~2.5%, что менее эффективно чем метод замены на среднее значение и удаления выбросов
# На метод случайного леса замена значений никак не повлияла, как в случае без настройки гиперпараметров, так и с настройкой