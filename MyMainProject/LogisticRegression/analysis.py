import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.covariance import EllipticEnvelope 
from sklearn.datasets import make_blobs

#Чтобы вывод был только в этом файле (не в тех, в которые он импортирован)
def aprint(*args, **kwargs):
    if __name__ == "__main__":
        print(*args, **kwargs)

df = pd.read_excel('C:/MyProject/glass.xlsx')

#===============\
#АНАЛИЗ ДАТАСЕТА|
#---------------/

aprint("ВЫВОД ДАТАФРЕЙМА:\n")
aprint(df)
aprint("\n" * 3)

aprint("ТИП СТОЛБЦОВ:\n")
aprint(df.dtypes)
aprint("\n" * 3)

aprint("ИНФОРМАЦИЯ О ДАТАФРЕЙМА:\n")
aprint(df.info())
aprint("\n" * 3)

aprint("ОПИСАНИЕ ДАТАФРЕЙМА:\n")
aprint(df.describe)
aprint("\n" * 3)

# Проверка на то, что в датасете имеются null значения      
has_null = False
for header in df.columns:
    if df[header].isnull().any():
        has_null = True
        break
if (has_null):
    aprint("Датасет содержит null значение.\n")
else:
    aprint("Датасет не содержит null значений.\n")
aprint("\n" * 3)

aprint("КОЛИЧЕСТВО РАЗЛИЧНЫХ ЗНАЧЕНИЙ \'CLASS\':")
aprint(df['Class'].value_counts())
aprint("\n" * 3)

# Сгруппировать строки по значениям столбца 'Class', вычислить среднее
# каждой группы
aprint("ГРУППИРОВКА ДАТАФРЕЙМА:\n")
aprint(df.groupby('Class').mean())
aprint("\n" * 3)

#==================================\
#ПРЕОБРАЗОВАНИЕ В МАТРИЦУ ПРИЗНАКОВ|
#----------------------------------/

# Удалить дубликаты
df = df.drop_duplicates()

# Матрица признаков (отбрасываем последний столбец с "классификацией"
# вид: "столбец" <=> "признак")
features = df.drop('Class', axis=1).to_numpy()

#=========================\
# СТАНДАРТИЗАЦИЯ ПРИЗНАКОВ|
#-------------------------/

# Создание шкалировщика и стандартизации признаков
standart_scaler = preprocessing.StandardScaler()
standardized_features = standart_scaler.fit_transform(features)

aprint("СТАНДАРТИЗИРОВАННЫЙ ПЕРВЫЙ ПРИЗНАК:")
aprint(standardized_features[:, 0])
aprint("Среднее:", round(standardized_features.mean()))
aprint("Стандартное отклонение:", standardized_features.std())
aprint("\n" * 3)

#=====================\
# ОБНАРУЖЕНИЕ ВЫБРОСОВ|
#---------------------/

# Создаём детектор и выполняем его подгонку
aprint("ОБНАРУЖЕНИЕ ВЫБРОСОВ:")
aprint("(№: номер наблюдения)\n")

# Метод Z-оценки(стандартной оценки) (корректно работает для >12 наблюдений)
# В её основе лежит сопоставление данных с распределением, среднее значение которого = 0, а стандартное отклонение = 1.
# Смысл: После того, как мы центрировали и масштабировали данные, всё, что слишком далеко от нуля (threeshold = 3), следует считать выбросом.
def outliers_z_score(ys):
    threshold = 3 #Путём просмотра количества выбросов при различных "порогах" для обранужения выброса было выбрано значение 3

    #mean_y = np.mean(ys)
    #stdev_y = np.std(ys)
    mean_y = 0 #т.к. были изначально стандартизированы
    stdev_y = 1 #т.к. были изначально стандартизированы

    z_scores = [(y - mean_y) / stdev_y for y in ys]
    return np.where(np.abs(z_scores) > threshold)

for i in range(0, standardized_features.shape[1]):
    aprint(i+1, ": ", outliers_z_score(standardized_features[:, i])[0])
aprint("\n" * 3)

outliers_in_6 = outliers_z_score(standardized_features[:, 5])
aprint("Выбросы в 6 признаке:", outliers_in_6[0])
aprint("\nИх стандартизированные значения:")
for i in outliers_in_6[0]:
    aprint(standardized_features[i, 5])
aprint("\nНормальные стандартизированные значения (не выбросы):")
aprint(standardized_features[:10, 5])

outliers_in_8 = outliers_z_score(standardized_features[:, 7])
aprint("\n\n\nВыбросы в 8 признаке:", outliers_in_8[0])
aprint("\nИх стандартизированные значения:")
for i in outliers_in_8[0]:
    aprint(standardized_features[i, 7])
aprint("\nНормальные стандартизированные значения (не выбросы):")
aprint(standardized_features[:10, 7])

outliers_in_9 = outliers_z_score(standardized_features[:, 8])
aprint("\n\n\nВыбросы в 9 признаке:", outliers_in_9[0])
aprint("\nИх стандартизированные значения:")
for i in outliers_in_9[0]:
    aprint(standardized_features[i, 8])
aprint("\nНормальные стандартизированные значения (не выбросы):")
aprint(standardized_features[:10, 8])

# Функция сравнения двух стоблцов
#def compare_columns(df, column1, column2):
#    match_count = 0
#    mismatches = []
#
#    for index, row in df.iterrows():
#        if row[column1] == row[column2]:
#            match_count += 1
#        else:
#            mismatches.append(index)
#    return match_count, mismatches