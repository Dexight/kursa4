import analysis as a
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

print("\n" * 3)

# Генерация цветов и возврат списка
def getColors(n):
    colors = []
    cm = plt.cm.get_cmap('hsv', n)
    for i in np.arange(n):
        colors.append(cm(i))
    return colors
df = a.df

plt.figure(num=1, figsize=(6, 6))
df['Class'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.show()

# Количество ненулевых значений
not_null_counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
headers = df.columns

def counter(row):
    for i in range(headers.size):
        if row[headers[i]] != 0:
            not_null_counts[i] += 1
df.apply(counter, axis = 1)

plt.bar(np.arange(headers.size), not_null_counts, color=getColors(headers.size))
plt.xticks(np.arange(headers.size), headers, rotation=90, fontsize=14)
plt.ylabel('Количество', fontsize=14)
plt.yticks(fontsize=14)
plt.title('Количество ненулевых наблюдений', fontsize=14)
plt.show()

# Максимальные значения признаков
unique_values = df['Class'].unique()
df['Class'].replace({unique_values[0]:0, unique_values[1]:1},inplace=True)
max_values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def get_max(row):
    global max_values
    for i in range(len(headers)):
        if row[i] > max_values[i]:
            max_values[i] = row[i]
df.apply(get_max, axis = 1)

plt.bar(np.arange(headers.size), max_values, color=getColors(headers.size))
plt.xticks(np.arange(headers.size), headers, rotation=90, fontsize=14)
plt.ylabel('Значение', fontsize=14)
plt.yticks(fontsize=14)
plt.title('Максимальные значения признаков', fontsize=14)
plt.show()

#Тепловая карта корелляции между элементами
correlation_matrix = df.corr()
print(df)
plt.figure(figsize=(df.shape[0], df.shape[1]))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Карта корелляции')
plt.show()

#Построение диаграмм разброса на основе тепловой карты
df_matrix = df.to_numpy()
figure, (plt1, plt2, plt3, plt4) = plt.subplots(1, 4, figsize=(12, 5))

# Диаграмма разброса для пары (Mg, Al)
plt1.scatter(df_matrix[:, 2], df_matrix[:, 3])
plt1.set_title('Разброс для (Mg, Al)')
plt1.set_xlabel('Mg')
plt1.set_ylabel('Al')

# Диаграмма разброса для пары (Mg, Ba)
plt2.scatter(df_matrix[:, 2], df_matrix[:, 7])
plt2.set_title('Разброс для (Mg, Ba)')
plt2.set_xlabel('Mg')
plt2.set_ylabel('Ba')

# Диаграмма разброса для пары (Mg, Class)
plt3.scatter(df_matrix[:, 2], df_matrix[:, 9])
plt3.set_title('Разброс для (Mg, Class)')
plt3.set_xlabel('Mg')
plt3.set_ylabel('Class')

# Диаграмма разброса для пары (Ba, Class)
plt4.scatter(df_matrix[:, 7], df_matrix[:, 9])
plt4.set_title('Разброс для (Ba, Class)')
plt4.set_xlabel('Ba')
plt4.set_ylabel('Class')

plt.tight_layout()
plt.show()