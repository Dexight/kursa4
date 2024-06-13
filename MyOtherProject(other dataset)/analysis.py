import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.covariance import EllipticEnvelope 
from sklearn.datasets import make_blobs

#Чтобы вывод был только в этом файле (не в тех, в которые он импортирован)
def aprint(*args, **kwargs):
    if __name__ == "__main__":
        print(*args, **kwargs)

df = pd.read_csv('C:/MyProject2/glass2.csv')

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

aprint("КОЛИЧЕСТВО РАЗЛИЧНЫХ ЗНАЧЕНИЙ \'Type\':")
aprint(df['Type'].value_counts())
aprint("\n" * 3)

# Сгруппировать строки по значениям столбца 'Type', вычислить среднее
# каждой группы
aprint("ГРУППИРОВКА ДАТАФРЕЙМА:\n")
aprint(df.groupby('Type').mean())
aprint("\n" * 3)

#==================================\
#ПРЕОБРАЗОВАНИЕ В МАТРИЦУ ПРИЗНАКОВ|
#----------------------------------/

# Удалить дубликаты
df = df.drop_duplicates()

# Матрица признаков (отбрасываем последний столбец с "классификацией"
# вид: "столбец" <=> "признак")
features = df.drop('Type', axis=1).to_numpy()

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

# Вывод: Думаю, что обработка выбросов здесь не нужна, т.к. объектов классов и так не много, и это может повлиять на обучение моделей