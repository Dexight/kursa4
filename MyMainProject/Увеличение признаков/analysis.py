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
features = df[[' Mg', ' Ba', ' Al', ' K']].to_numpy()

#=========================\
# СТАНДАРТИЗАЦИЯ ПРИЗНАКОВ|
#-------------------------/

# Создание шкалировщика и стандартизации признаков
standart_scaler = preprocessing.StandardScaler()
standardized_features = standart_scaler.fit_transform(features)
aprint("Среднее:", round(standardized_features.mean()))
aprint("Стандартное отклонение:", standardized_features.std())
print(features)

from sklearn.preprocessing import PolynomialFeatures, scale
standardized_features = PolynomialFeatures(include_bias=False).fit_transform(scale(standardized_features))

print(standardized_features.shape)