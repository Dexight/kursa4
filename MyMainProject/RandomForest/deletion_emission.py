#Файл с обработкой выбросов
from analysis import * 

print("\n ПЕРВОНАЧАЛЬНЫЕ ДАННЫЕ МЕТОДОВ КЛАССИФИКАЦИИ: ")

from RandomForest import RandForest

print("\n" * 2)

#1) Удаление выбросов
print("ОБРАБОТКА МЕТОДОМ УДАЛЕНИЯ ВЫБРОСОВ:")
print("\n" * 2)

# Объединяем индексы выбросов
outliers_indices = []
outliers_indices.extend(outliers_in_6[0])
outliers_indices.extend(outliers_in_8[0])
outliers_indices.extend(outliers_in_9[0])

print(outliers_indices)

# Удаляем дубликаты индексов
outliers_indices = list(set(outliers_indices))
print(outliers_indices)

# Удаление строк с выбросами
cleaned_df = np.delete(df['Class'], outliers_indices, axis=0)
cleaned_features = np.delete(standardized_features, outliers_indices, axis=0)

# Проверка размерности после удаления выбросов
print("Изначальная размерность:", standardized_features.shape)
print("Размерность после удаления выбросов:", cleaned_features.shape)

RandForest(cleaned_features, cleaned_df)

# Вывод: После удаления выбросов результаты стремятся к 100% во всех методах. 