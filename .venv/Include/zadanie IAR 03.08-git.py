 # #1. Задание для итоговой аттестации.
# #Для выполнения итоговой аттестационной работы с учетом всех требований:
# 1.	Программа реализована на языке программирования Python.
# 2.	Выполнена загрузка и чтение данных из файлов.
# 3.	Выполнена предварительная обработка данных (очистка и форматирование данных).
# в осоновном удаление пропущенных значений
# 4.	Применены методы математической статистики для обработки данных.
# 5.	Выполнен поиск закономерностей в данных.
# 6.	Выполнена визуализация данных.
# 7.	Составлена гипотеза о данных и выполнена проверка соответствующей гипотезы.
# 8.	Полученные результаты интерпретированы в соответствии с поставленной бизнес-задачей.
# 9.	Подготовлен и опубликован дашборд.
#Задача: выяснить, какие из признаков оказывают большое влияние на ценообразование квартир.

# airports_nearest		расстояние до ближайшего аэропорта в метрах (м)
# balcony				число балконов
# ceiling_height			высота потолков (м)
# cityCenters_nearest		расстояние до центра города (м)
# floor				этаж
# floors_total			всего этажей в доме
# is_apartment			апартаменты (булев тип)
# kitchen_area			площадь кухни в квадратных метрах (м²)
# last_price			цена на момент снятия с публикации
# living_area			жилая площадь в квадратных метрах(м²)
# open_plan			свободная планировка (булев тип)
# parks_around3000		число парков в радиусе 3 км
# parks_nearest			расстояние до ближайшего парка (м)
# ponds_around3000		число водоёмов в радиусе 3 км
# ponds_nearest			расстояние до ближайшего водоёма (м)
# rooms				число комнат
# studio				квартира-студия (булев тип)
# total_area			площадь квартиры в квадратных метрах (м²)


import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sb
import scipy.stats as st
import numpy as np
#
# # 2.Выполнена загрузка и чтение данных из файлов.
# # Загрузка и чтение данных
df = pd.read_csv('home_price.csv')

# Проверка данных:
# o Просмотрите первые 10 строк данных для понимания  структуры DataFrame.

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
print(df.head(10))


# Общая информация о данных:
print(df.info())

# 3.Выполнена предварительная обработка данных (очистка и форматирование данных).
# в осоновном удаление пропущенных значений

#3.1 Колонка is_apartment' тип данных object, заменим на bool
df = df.astype({'is_apartment': 'bool'},
               errors='ignore')  # errors='ignore' не пытается угадать кодировку—она игнорирует ошибки при декодировании, используя кодировку, которую вы явно указали в вызове
df.info()

# 3.2 проверим на дубликаты
# Выведем столбцы с дубликатами для ознакомления
duplicates_rows = df.loc[df.duplicated()]
print("\nDuplicated rows:")
print(duplicates_rows)
#удалим дубликаты и запишем df в df_cleaned
df_cleaned = df.drop_duplicates()
print("\nDataFrame after removing duplicates:")
print(df_cleaned.info())


#3.3 Определим количество пропущенных значений в каждом столбце. и аномальные значения
print(df_cleaned.isnull().sum())
# last_price                 0
# total_area                 0
# rooms                      0
# ceiling_height          9195
# floors_total              86
# living_area             1903
# floor                      0
# is_apartment               0
# studio                     0
# open_plan                  0
# kitchen_area            2277
# balcony                11518
# airports_nearest        5541
# cityCenters_nearest     5518
# parks_around3000        5517
# parks_nearest          15619
# ponds_around3000        5517
# ponds_nearest          14588

#
# #Видно, что пропущенных значений достаточно много,
# #Также с помощью describe(методов описательной статистики) выясним есть ли аномалии
summary_stats = df_cleaned.describe()
print(summary_stats)
# #          last_price    total_area         rooms  ceiling_height  floors_total  \
# # count  2.369800e+04  23698.000000  23698.000000    14503.000000  23612.000000
# # mean   6.541718e+06     60.349404      2.070681        2.771513     10.673513
# # std    1.088721e+07     35.654647      1.078405        1.261098      6.597140
# # min    1.219000e+04     12.000000      0.000000        1.000000      1.000000
# # 25%    3.400000e+06     40.000000      1.000000        2.520000      5.000000
# # 50%    4.650000e+06     52.000000      2.000000        2.650000      9.000000
# # 75%    6.800000e+06     69.900000      3.000000        2.800000     16.000000
# # max    7.630000e+08    900.000000     19.000000      100.000000     60.000000
# #
# #         living_area         floor  kitchen_area       balcony  \
# # count  21795.000000  23698.000000  21421.000000  12180.000000
# # mean      34.458525      5.892312     10.569807      1.150082
# # std       22.030727      4.885347      5.905438      1.071300
# # min        2.000000      1.000000      1.300000      0.000000
# # 25%       18.600000      2.000000      7.000000      0.000000
# # 50%       30.000000      4.000000      9.100000      1.000000
# # 75%       42.300000      8.000000     12.000000      2.000000
# # max      409.700000     33.000000    112.000000      5.000000
# #
# #        airports_nearest  cityCenters_nearest  parks_around3000  parks_nearest  \
# # count      18157.000000         18180.000000      18181.000000    8079.000000
# # mean       28793.672193         14191.277833          0.611408     490.804555
# # std        12630.880622          8608.386210          0.802074     342.317995
# # min            0.000000           181.000000          0.000000       1.000000
# # 25%        18585.000000          9238.000000          0.000000     288.000000
# # 50%        26726.000000         13098.500000          0.000000     455.000000
# # 75%        37273.000000         16293.000000          1.000000     612.000000
# # max        84869.000000         65968.000000          3.000000    3190.000000
# #
# #        ponds_around3000  ponds_nearest
# # count      18181.000000    9110.000000
# # mean           0.770255     517.980900
# # std            0.938346     277.720643
# # min            0.000000      13.000000
# # 25%            0.000000     294.000000
# # 50%            1.000000     502.000000
# # 75%            1.000000     729.000000
# # max            3.000000    1344.000000
#
# # Из основных показателей описательной статистики делаем вывод, что аномалии есть практически в каждом столбце.
# # Например: last_price
# # mean: 6,541,718
# # median (50%): 4,650,000
# # std: 10,887,210
# # min: 12,190
# # max: 763,000,000
# # Анализ:
# # Среднее значение значительно выше медианы, что указывает на наличие высоких выбросов.
# # Стандартное отклонение очень велико, что также подтверждает наличие высоких значений.
# # Максимальное значение (763 млн) далеко выходит за пределы 75% перцентиля (6.8 млн), что свидетельствует о наличии экстремальных выбросов
#
# # Переменные, такие как last_price, total_area, rooms, ceiling_height, floors_total, living_area, floor, kitchen_area, airports_nearest, cityCenters_nearest, parks_nearest, и ponds_nearest имеют значительные выбросы.
#
#
#
# #для начала удалим пропуске в столбце # floors_total , тк их всего 86
df_cleaned = df_cleaned.dropna(subset='floors_total')
print(df_cleaned['floors_total'].isnull().sum()) # проверим удалились ли значения, да
#
#Колонки категорий помещений -is_apartment, open_plan, studio, тип bool, проверим все ли строки заполнены, а
#пропущенные заменим на моду по трем
# # Подсчет количества квартир по типам is_apartment, studio, open_plan
type_counts = df_cleaned.groupby(['is_apartment', 'studio', 'open_plan']).size()

# Подсчет количества квартир с непроставленным типом (где есть NaN)
missing_type_counts = df_cleaned[['is_apartment', 'studio', 'open_plan']].isnull().any(axis=1).sum()

# Подсчет количества квартир, где все 3 типа равны False
all_false_count = len(df_cleaned[(df_cleaned['is_apartment'] == False) &
                         (df_cleaned['studio'] == False) &
                         (df_cleaned['open_plan'] == False)])

# Вывод результатов
print("Количество квартир по типам is_apartment, studio, open_plan:")
print(type_counts)

print("\nКоличество квартир без проставленного типа (где есть NaN):")
print(missing_type_counts)

print("\nКоличество квартир, где все 3 типа равны False (is_apartment, studio, open_plan):")
print(all_false_count)


# Найти моду для каждой из категорий
mode_is_apartment = df_cleaned['is_apartment'].mode()[0]
mode_studio = df_cleaned['studio'].mode()[0]
mode_open_plan = df_cleaned['open_plan'].mode()[0]

# Найти строки, где все три категории равны False
all_false_indices = df_cleaned[(df_cleaned['is_apartment'] == False) &
                       (df_cleaned['studio'] == False) &
                       (df_cleaned['open_plan'] == False)].index
print("Количество квартир, где все 3 типа равны False:")
print(all_false_indices)
# Заменить значения для этих строк
df_cleaned.loc[all_false_indices, 'is_apartment'] = mode_is_apartment
df_cleaned.loc[all_false_indices, 'studio'] = mode_studio
df_cleaned.loc[all_false_indices, 'open_plan'] = mode_open_plan

# Проверка, что после замены нет строк, где все три категории равны False
all_false_count_after = len(df_cleaned[(df_cleaned['is_apartment'] == False) &
                               (df_cleaned['studio'] == False) &
                               (df_cleaned['open_plan'] == False)])

print("Количество квартир, где все 3 типа равны False после замены:")
print(all_false_count_after)


# #для столбца 'ceiling_height' (высота потолков) мы заменим пропущенные значения на медиану(2,65),
# #посчитаем медиану
median_ceiling_height = float(df_cleaned['ceiling_height'].median())
print("Медиана высоты потолков:", median_ceiling_height)
#
# #Заменяем значения высоты потолков меньше 2,0 м на медиану
df_cleaned['ceiling_height'] = np.where(df_cleaned['ceiling_height'] < 2.45, median_ceiling_height, df_cleaned['ceiling_height'])
#
# #Заменяем значения высоты потолков больше 5,0 метров на медиану
df_cleaned['ceiling_height'] = np.where(df_cleaned['ceiling_height'] > 5.0, median_ceiling_height, df_cleaned['ceiling_height'])
#  #Заменяем пустые значения высоты потолков на медиану
df_cleaned['ceiling_height'] = df_cleaned['ceiling_height'].fillna(median_ceiling_height)
#

# # Для остальных столбцов построим boxplot() - 'ящик с усами' по каждому столбцу чтобы проанализировать аномалии в столбцах
#
#Дополнительно функция для расчета значений нижнего и верхнего 'усов' (для графиков и фильтрации)

def calc_boxplot(df_cleaned_col : pd.Series) -> tuple:
    Q1, median, Q3 = np.percentile(np.asarray(df_cleaned_col.dropna()), [25, 50, 75])
    IQR = Q3 - Q1
    loval = Q1 - 1.5 * IQR
    hival = Q3 + 1.5 * IQR
    wiskhi = np.compress(np.asarray(df_cleaned_col.dropna()) <= hival, np.asarray(df_cleaned_col.dropna()))
    wisklo = np.compress(np.asarray(df_cleaned_col.dropna()) >= loval, np.asarray(df_cleaned_col.dropna()))
    actual_hival = np.max(wiskhi)
    actual_loval = np.min(wisklo)
    return actual_loval, actual_hival


cols = [
    'last_price', 'total_area', 'rooms', 'ceiling_height',
    'floors_total', 'living_area', 'floor', 'kitchen_area', 'balcony',
    'airports_nearest', 'cityCenters_nearest', 'parks_nearest', 'ponds_nearest','parks_around3000', 'ponds_around3000']


#Выполняем визуализацию
plt.subplots(15, figsize = (24, 150))
for index_fig, col in enumerate(cols, start = 0):
    plt.subplot(15, 2, 2*index_fig + 1)
    ax = sb.boxplot(data = df,
                   y = col)
    actual_loval, actual_hival = calc_boxplot(df[col])
    ax.axhline(actual_loval,
            color = 'blue',
            label = f'Нижний ус: {actual_loval}')
    ax.axhline(actual_hival,
            color = 'red',
            label = f'Верхний ус: {actual_hival}')
    q_1 = np.percentile(df[col].dropna(), 1)
    ax.axhline(q_1,
            color = 'pink',
            label = f'1-я перцинтель: {q_1:2f}')
    q_99 = np.percentile(df[col].dropna(), 99)
    ax.axhline(q_99,
            color = 'green',
            label = f'99-я перцинтель: {q_99:2f}')
    plt.legend()

    plt.subplot(15, 2, 2*index_fig + 2)
    ax = sb.histplot(df[col]);
    ax.axvline(actual_loval,
            color = 'blue',
            label = f'Нижний ус: {actual_loval}')
    ax.axvline(actual_hival,
            color = 'red',
            label = f'Верхний ус: {actual_hival}')
    q_1 = np.percentile(df[col].dropna(), 1)
    ax.axvline(q_1,
            color = 'pink',
            label = f'1-я перцинтель: {q_1:2f}')
    q_99 = np.percentile(df[col].dropna(), 99)
    ax.axvline(q_99,
            color = 'green',
            label = f'99-я перцинтель: {q_99:2f}')

    plt.legend()
plt.savefig('boxplots_before.png')









#Устраняем аномалии

# # Уберем аномалии в остальных столбцах, для того чтобы заполнить пропушенные значения средним или медианой из боксплота видно что в основном аномалии по верхней границе
# #Точно можно удалить аномалии в:
# last_price,total_area,ceiling_height,living_area,kitchen_area,airports_nearest,cityCenters_nearest, parks_nearest, ponds_nearest,rooms,floors_total, floor

#создаем фильтр для удаления аномалий,
filt = ((df_cleaned.last_price.between(np.nanpercentile(df_cleaned.last_price, 1),
                                       np.nanpercentile(df_cleaned.last_price, 99))) #для last price исходя из графика, задаем нижнюю границу 1 процентиль и верхнюю границу нашей функции calc_boxplot
      & (df_cleaned.total_area.between(np.nanpercentile(df_cleaned.total_area, 1),
                                       np.nanpercentile(df_cleaned.total_area, 99))) #для total_area задаем от 1 до 99 процентиля
      & ((df_cleaned.ceiling_height.between(df_cleaned.ceiling_height.min(),
                                   df_cleaned.ceiling_height.max())) ) #для ceiling_height от мин 2.45 до макс значения
      & ((df_cleaned.living_area.between(np.nanpercentile(df_cleaned.living_area, 1),
                                calc_boxplot(df_cleaned.living_area)[1])) | (df_cleaned.living_area.isna())) #для living_area между 1 проц и верхн усом
      & ((df_cleaned.kitchen_area.between(np.nanpercentile(df_cleaned.kitchen_area, 1),
                                calc_boxplot(df_cleaned.kitchen_area)[1])) | (df_cleaned.kitchen_area.isna()))
      & ((df_cleaned.airports_nearest.between(np.nanpercentile(df_cleaned.airports_nearest, 1),
                                     np.nanpercentile(df_cleaned.airports_nearest, 99))) | (df_cleaned.airports_nearest.isna()))
      & ((df_cleaned.cityCenters_nearest.between(np.nanpercentile(df_cleaned.cityCenters_nearest, 1),
                                         39000)) | (df_cleaned.cityCenters_nearest.isna()))
      & ((df_cleaned.parks_nearest.between(calc_boxplot(df_cleaned.parks_nearest)[0],
                                  calc_boxplot(df_cleaned.parks_nearest)[1])) | (df_cleaned.parks_nearest.isna()))
      & ((df_cleaned.ponds_nearest.between(calc_boxplot(df_cleaned.ponds_nearest)[0],
                                  np.nanpercentile(df_cleaned.ponds_nearest, 99))) | (df_cleaned.ponds_nearest.isna()))
      & (df_cleaned.rooms.between(np.nanpercentile(df_cleaned.rooms, 1),
                          calc_boxplot(df_cleaned.rooms)[1]))
      & ((df_cleaned.floors_total.between(np.nanpercentile(df_cleaned.floors_total, 1),
                                  np.nanpercentile(df_cleaned.floors_total, 99))) | (df_cleaned.floors_total.isna()) )
      & (df_cleaned.floor.between(np.nanpercentile(df_cleaned.floor, 1),
                           np.nanpercentile(df.floor, 99)))
      )

print(f"После удаления аномальных значений осталось {(df_cleaned[filt].shape[0] / df_cleaned.shape[0]):.2%} данных.")

#Также посмотрим описательные статистики отфильтрованного датафрейма
df_cleaned[filt].describe()

#Потеряли менее 20%, результат допустимый, но при желании с фильтрами можно ещё поэкспериментировать

#Сохраняем изменения
df_before = df_cleaned.copy()
df_cleaned=df_cleaned[filt]
#Сделаем итоговый график boxplot до и после удаления аномальных значений для наглядности
df_after = df_cleaned[filt]

# Определите, какие колонки вы хотите сравнивать
columns_to_plot = [
    'last_price', 'total_area', 'ceiling_height', 'living_area', 'kitchen_area',
    'airports_nearest', 'cityCenters_nearest', 'parks_nearest', 'ponds_nearest',
    'rooms', 'floors_total', 'floor'
]

# Построение графиков boxplot
plt.figure(figsize=(16, 20))

for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(len(columns_to_plot), 2, 2*i-1)
    sb.boxplot(y=df_before[column], color='lightblue')
    plt.title(f'{column} - Before')

    plt.subplot(len(columns_to_plot), 2, 2*i)
    sb.boxplot(y=df_after[column], color='lightgreen')
    plt.title(f'{column} - After')

plt.tight_layout()
# Сохранение графика в файл
plt.savefig('boxplots_before_after.png')
plt.show()

#
# Проверим удалились ли аномалии

print("Статистика после удаления аномалий:")
print(df_cleaned.describe())
print(df_cleaned.isnull().sum())
print(df_cleaned.info())

# #Пропущенные значения в остальных столбцах заменим на медиану или средние значения либо удалим

# #заменим медианой по группе комнат пропуски жилой площади 'living_area' и площади кухни 'kitchen_area'
# #заполнение пропущенных значений в столбце living_area медианными значениями этого столбца, сгруппированными по количеству комнат (rooms)
transform_living_area = df_cleaned.groupby(by='rooms')['living_area'].transform('median')
#
# # ['living_area']: Из каждой группы выбирается столбец living_area.
# # .transform('median'): Для каждого значения в столбце living_area в соответствующей группе (по количеству комнат) вычисляется медиана. Метод transform возвращает серию, которая имеет ту же длину, что и исходный DataFrame, где для каждой строки будет записана медиана соответствующей группы.
# # Результат этой операции — серия transform_living_area, где каждому значению в living_area сопоставляется медианное значение для той же группы по количеству комнат.
df_cleaned['living_area'] = df_cleaned['living_area'].fillna(transform_living_area)
#
# # .fillna(transform_living_area): Заполняет пропущенные значения (NaN) в столбце living_area значениями из серии transform_living_area. То есть, если в столбце living_area есть пропущенное значение, оно заменяется на медиану living_area для соответствующего количества комнат (rooms).

##также заменим медианой по группе комнат пропуски площади кухни 'kitchen_area'
transform_kitchen_area = df_cleaned.groupby(by='rooms')['kitchen_area'].transform('median')
df_cleaned['kitchen_area'] = df_cleaned['kitchen_area'].fillna(transform_kitchen_area)

# #заменим оставшиеся пропуски 'kitchen_area' (190), которые относятся к студиям на ноль
df_cleaned['kitchen_area'] = df_cleaned['kitchen_area'].fillna(0)
# print(df_cleaned['kitchen_area'].isnull().sum()) # проверим удалились ли значения, да
# print(df_cleaned.isnull().sum()


#заменим   на медиану #Также пропуски в: балконах, расстояние до парков, водоёмов, центра и аэропортов заменим на медиану.
median_balcony = float(df_cleaned['balcony'].median())
print("Медиана balcony:", median_balcony)
#Заменяем пустые значения balcony на медиану
df_cleaned['balcony'] = df_cleaned['balcony'].fillna(median_balcony)
median_parks_nearest = float(df_cleaned['parks_nearest'].median())
print("Медиана parks_nearest:", median_parks_nearest)
df_cleaned['parks_nearest'] = df_cleaned['parks_nearest'].fillna(median_parks_nearest)
median_parks_around3000  = float(df_cleaned['parks_around3000'].median())
print("Медиана parks_around3000:", median_parks_around3000)
df_cleaned['parks_around3000'] = df_cleaned['parks_around3000'].fillna(median_parks_around3000)

median_ponds_nearest = float(df_cleaned['ponds_nearest'].median())
print("Медиана ponds_nearest:", median_ponds_nearest)
df_cleaned['ponds_nearest'] = df_cleaned['ponds_nearest'].fillna(median_ponds_nearest)


median_ponds_around3000 = float(df_cleaned['ponds_around3000'].median())
print("Медиана ponds_around3000:", median_ponds_around3000)
df_cleaned['ponds_around3000'] = df_cleaned['ponds_around3000'].fillna(median_ponds_around3000)

median_airports_nearest = float(df_cleaned['airports_nearest'].median())
print("Медиана airports_nearest:", median_airports_nearest)
df_cleaned['airports_nearest'] = df_cleaned['airports_nearest'].fillna(median_airports_nearest)

median_cityCenters_nearest = float(df_cleaned['cityCenters_nearest'].median())
print("Медиана cityCenters_nearest:", median_cityCenters_nearest)
df_cleaned['cityCenters_nearest'] = df_cleaned['cityCenters_nearest'].fillna(median_cityCenters_nearest)

print(df_cleaned.isnull().sum())
print(df_cleaned.info())
# #
# # #сохраним очищенный датасет df_cleaned
# df_cleaned.to_csv('df_cleaned1.csv', index=False)

# 4.Применены методы математической статистики для обработки данных.
# # # # Загрузка и чтение данных
df = pd.read_csv('df_cleaned.csv')

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
print(df.head(30))

print(df.isnull().sum())
print(df.info())
print(df.describe())
#          last_price    total_area         rooms  ceiling_height  floors_total  \
# count  1.991200e+04  19912.000000  19912.000000    19912.000000  19912.000000
# mean   5.292131e+06     55.044444      1.991714        2.677251     10.565137
# std    3.220813e+06     20.065820      0.898484        0.184956      6.273429
# min    1.000000e+06     25.360000      1.000000        2.450000      2.000000
# 25%    3.400000e+06     40.000000      1.000000        2.600000      5.000000
# 50%    4.500000e+06     50.700000      2.000000        2.650000      9.000000
# 75%    6.200000e+06     65.500000      3.000000        2.700000     15.000000
# max    3.600000e+07    197.000000      6.000000        4.800000     26.000000
#
#         living_area         floor  kitchen_area       balcony  \
# count  19912.000000  19912.000000  19912.000000  19912.000000
# mean      31.527642      5.724387      9.358521      1.078395
# std       13.404377      4.564624      2.879248      0.766253
# min       13.000000      1.000000      5.000000      0.000000
# 25%       18.400000      2.000000      7.000000      1.000000
# 50%       30.000000      4.000000      9.000000      1.000000
# 75%       41.000000      8.000000     11.000000      1.000000
# max       77.800000     23.000000     19.500000      5.000000
#
#        airports_nearest  cityCenters_nearest  parks_around3000  parks_nearest  \
# count      19912.000000         19912.000000      19912.000000   19912.000000
# mean       28125.743823         13892.333819          0.424317     443.967256
# std        10518.295944          6161.148104          0.709321     120.551307
# min         9522.000000          1327.000000          0.000000       1.000000
# 25%        20886.250000         11448.500000          0.000000     444.000000
# 50%        26943.000000         13279.000000          0.000000     444.000000
# 75%        34103.000000         15228.000000          1.000000     444.000000
# max        58523.000000         38868.000000          3.000000    1098.000000
#
#        ponds_around3000  ponds_nearest
# count      19912.000000   19912.000000
# mean           0.521344     527.489504
# std            0.819350     159.438897
# min            0.000000      20.000000
# 25%            0.000000     525.000000
# 50%            0.000000     525.000000
# 75%            1.000000     525.000000
# max            3.000000    1159.000000
#
#
# На основе предоставленной описательной статистики можно сделать следующие выводы о данных:
#
# 1. Стоимость недвижимости (last_price):
# Средняя стоимость составляет 5.292 миллиона рублей, с медианой в 4.5 миллиона рублей.
# Стандартное отклонение достаточно велико (3.22 миллиона рублей), что указывает на значительную вариативность цен.
# Минимальная стоимость — 1 миллион рублей, максимальная — 36 миллионов рублей, что подтверждает наличие объектов как низкого, так и высокого ценового диапазона.
# 25-й и 75-й перцентили (3.4 и 6.2 миллиона рублей) показывают, что большая часть объектов находится в диапазоне от 3.4 до 6.2 миллионов рублей.
# 2. Общая площадь (total_area):
# Средняя площадь составляет 55 кв.м., с медианой 50.7 кв.м.
# Стандартное отклонение в 20 кв.м. указывает на значительное расхождение в размерах квартир.
# Минимальная площадь — 25.36 кв.м., максимальная — 197 кв.м.
# 25-й и 75-й перцентили (40 и 65.5 кв.м.) показывают, что большинство квартир имеют площадь от 40 до 65.5 кв.м.
# 3. Количество комнат (rooms):
# Среднее количество комнат — 1.99, с медианой 2 комнаты.
# Стандартное отклонение — 0.898, что указывает на наличие значительного числа квартир с 1 или 3 комнатами.
# Минимальное количество комнат — 1, максимальное — 6, что соответствует распространенным типам квартир.
# 4. Высота потолков (ceiling_height):
# Средняя высота потолков — 2.68 м, с медианой 2.65 м.
# Стандартное отклонение — 0.185 м, что указывает на небольшую вариативность, но с максимальной высотой до 4.8 м, что может свидетельствовать о наличии редких квартир с высокими потолками.
# 5. Общая этажность здания (floors_total):
# Средняя этажность составляет 10.57 этажей.
# Стандартное отклонение — 6.27, что указывает на большой разброс высот зданий.
# Минимальная этажность — 2 этажа, максимальная — 26 этажей, что отражает разнообразие типов зданий в выборке.
# 6. Жилая площадь (living_area):
# Средняя жилая площадь составляет 31.53 кв.м., с медианой 30 кв.м.
# Стандартное отклонение — 13.4 кв.м., что указывает на наличие значительных различий в размерах жилой площади.
# Минимальная площадь — 13 кв.м., максимальная — 77.8 кв.м.
# 7. Этаж квартиры (floor):
# Средний этаж квартиры — 5.72, с медианой на 4 этаже.
# Стандартное отклонение — 4.56, что указывает на значительное разнообразие по расположению квартиры в здании.
# Минимальный этаж — 1, максимальный — 23.
# 8. Площадь кухни (kitchen_area):
# Средняя площадь кухни составляет 9.36 кв.м., с медианой 9 кв.м.
# Стандартное отклонение — 2.88 кв.м., что указывает на достаточно ровное распределение значений.
# Минимальная площадь — 5 кв.м., максимальная — 19.5 кв.м.
# 9. Близость к аэропортам (airports_nearest):
# Среднее расстояние до аэропортов — 28.1 км.
# Стандартное отклонение — 10.5 км, что указывает на значительное разнообразие расстояний.
# Минимальное расстояние — 9.5 км, максимальное — 58.5 км.
# 10. Близость к центрам города (cityCenters_nearest):
# Среднее расстояние до центра — 13.89 км.
# Стандартное отклонение — 6.16 км, что указывает на широкую географию объектов.
# Минимальное расстояние — 1.3 км, максимальное — 38.8 км.
# 11. Наличие парков и прудов в радиусе 3000 метров (parks_around3000 и ponds_around3000):
# Большинство объектов не имеют парков и прудов в радиусе 3000 метров (медианы равны 0).
# Максимальные значения показывают, что некоторые объекты имеют до 3 парков и прудов в пределах 3000 метров.
# 12. Близость к паркам и прудам (parks_nearest и ponds_nearest):
# Среднее расстояние до ближайшего парка — 444 метра, до ближайшего пруда — 527 метров.
# Стандартное отклонение — около 120 и 159 метров соответственно.
# Минимальное расстояние — 1 метр для парков и 20 метров для прудов, максимальное — 1098 и 1159 метров.
# Общие выводы:
# Вариативность данных: Данные характеризуются значительной вариативностью, что проявляется в высоких значениях стандартного отклонения для многих параметров, таких как стоимость, площадь, высота потолков и расстояние до аэропортов и центров города. Это указывает на широкий диапазон характеристик недвижимости в выборке.
# Наличие выбросов: Большие различия между минимальными, максимальными значениями и перцентилями (например, стоимость и площадь) могут свидетельствовать о наличии выбросов в данных.
# Сбалансированность выборки: Медианы для большинства показателей находятся близко к средним значениям, что говорит о сбалансированности данных и отсутствии значительных перекосов.
# В целом, данные выглядят достаточно полными и репрезентативными, но возможны выбросы, которые следует учитывать при дальнейшем анализе.



# Сделаем расчет основных показателей статистики -среднее значение, стандартное отклонение и дисперсия для ['total_area']
# в зависимости от типа квартир - is_apartment , studio, open_plan

# Группировка данных по нескольким признакам: is_apartment, studio, open_plan
grouped = df.groupby(['is_apartment', 'studio', 'open_plan'])

# Расчет среднего значения, стандартного отклонения и дисперсии для каждой группы
mean_total_area = grouped['total_area'].mean()
std_total_area = grouped['total_area'].std()
var_total_area = grouped['total_area'].var()
# Вычисление корреляционной матрицы
#corr_matrix = df[columns].corr()

# Печать результатов
print("Среднее значение общей площади по типу квартир:")
print(mean_total_area)

print("\nСтандартное отклонение общей площади по типу квартир:")
print(std_total_area)

print("\nДисперсия общей площади по типу квартир:")
print(var_total_area)
#
# 5.	Выполнен поиск закономерностей в данных.
#Задача: выяснить, какие из признаков оказывают большое влияние на ценообразование квартир.
# В нашем датасете мы проверяем каие факторы влияют на стоимость недвижимости?
# и можно сделать гипотезу на такую то составляющую стоймости влияют такие то факторы (например квартиры с разной площадью и квартиры с разной высотой потолков стоят по разному )
# Пункт про закономерности можно сделать корреляц матрицу зависимость линейная или монтнотонная или скаттерплот график

#Корреляционная матрица
# Преобразование булевых столбцов в числовой формат
df['is_apartment'] = df['is_apartment'].astype(int)
df['open_plan'] = df['open_plan'].astype(int)
df['studio'] = df['studio'].astype(int)
print(df.info())

# Выбор столбцов для анализа
columns = [
    'airports_nearest',
    'balcony',
    'ceiling_height',
    'cityCenters_nearest',
    'floor',
    'floors_total',
    'is_apartment',
    'kitchen_area',
    'last_price',
    'living_area',
    'open_plan',
    'parks_around3000',
    'parks_nearest',
    'ponds_around3000',
    'ponds_nearest',
    'rooms',
    'studio',
    'total_area'
]
# Выбор нужных столбцов из DataFrame
df_subset = df[columns]

# Вычисление корреляционной матрицы
corr_matrix = df_subset.corr()

# Печать корреляционной матрицы
#print(corr_matrix)

# Вывод корреляций с last_price
print("\nКорреляции с last_price:")
print(corr_matrix['last_price'].sort_values(ascending=False))

# Корреляции с last_price:
# last_price             1.000000
# total_area             0.719583
# living_area            0.566810
# kitchen_area           0.468916
# rooms                  0.417236
# ceiling_height         0.369747
# ponds_around3000       0.280909
# parks_around3000       0.265162
# floors_total           0.147958
# floor                  0.115537
# balcony                0.087493
# airports_nearest       0.013389
# parks_nearest          0.009078
# is_apartment           0.008736
# open_plan              0.000952
# studio                -0.013957
# ponds_nearest         -0.045505
# cityCenters_nearest   -0.282269
# На основе корреляционной матрицы с last_price можно сделать следующие выводы:
#
# 1. Сильные и умеренные положительные корреляции:
# Общая площадь (total_area): Корреляция с ценой составляет 0.719, что указывает на сильную положительную связь. Это значит, что с увеличением площади квартиры цена также возрастает, что логично, так как большая площадь обычно подразумевает более высокую стоимость.
# Жилая площадь (living_area): Корреляция 0.567 также указывает на положительную связь, хотя и не такую сильную, как у общей площади. Это свидетельствует о том, что жилая площадь значимо влияет на цену, но меньшей мерой, чем общая площадь.
# Площадь кухни (kitchen_area): Корреляция 0.469 указывает на умеренную положительную связь. Большая кухня может повышать стоимость, особенно если она соответствует высоким стандартам и удобству.
# Количество комнат (rooms): Корреляция 0.417 также свидетельствует о положительной связи. Это указывает на то, что квартиры с большим числом комнат, как правило, дороже, хотя эта связь менее сильна, чем с общей площадью.
# Высота потолков (ceiling_height): Корреляция 0.370 показывает умеренную положительную связь, что подтверждает идею о том, что более высокие потолки могут быть связаны с более высокими ценами, возможно, из-за восприятия большего пространства и роскоши.
# 2. Слабые положительные корреляции:
# Количество прудов в радиусе 3000 метров (ponds_around3000): Корреляция 0.281 указывает на слабую положительную связь, что может говорить о том, что наличие прудов вблизи может слегка повышать цену недвижимости.
# Количество парков в радиусе 3000 метров (parks_around3000): Корреляция 0.265 также указывает на слабую положительную связь, предполагая, что наличие парков в окрестностях также может слегка увеличивать стоимость.
# Общая этажность здания (floors_total): Корреляция 0.148 является слабой положительной, что может свидетельствовать о незначительном влиянии высоты здания на стоимость недвижимости.
# Этаж квартиры (floor): Корреляция 0.116 также слабая, что указывает на небольшое влияние этажа на стоимость. Возможно, квартиры на средних этажах могут быть дороже, чем на первом или последнем.
# 3. Очень слабые положительные и отрицательные корреляции:
# Наличие балкона (balcony): Корреляция 0.087 показывает очень слабую положительную связь, что может означать, что наличие балкона не оказывает значительного влияния на цену.
# Близость к аэропортам (airports_nearest): Корреляция 0.013 практически отсутствует, что указывает на то, что расстояние до аэропорта не оказывает влияния на цену недвижимости.
# Близость к паркам (parks_nearest): Корреляция 0.009 также очень слабая, что говорит о том, что непосредственная близость к паркам незначительно влияет на стоимость.
# Наличие статуса апартаментов (is_apartment): Корреляция 0.009 указывает на минимальное влияние этого фактора на цену.
# Наличие открытой планировки (open_plan): Корреляция 0.001 указывает на отсутствие влияния на цену.
# Студийные квартиры (studio): Корреляция -0.014 также практически отсутствует, что свидетельствует о минимальном влиянии наличия студийного типа квартиры на стоимость.
# Расстояние до ближайшего пруда (ponds_nearest): Корреляция -0.046 указывает на слабую отрицательную связь, что может говорить о том, что чем ближе пруд, тем меньше может быть цена, возможно из-за возможных недостатков, таких как комары или затопляемость.
# Близость к центру города (cityCenters_nearest): Корреляция -0.282 указывает на слабую отрицательную связь, что необычно, так как обычно ближе к центру города недвижимость дороже. Это может свидетельствовать о том, что в данной выборке более удаленные от центра районы могут быть более престижными или в центре представлены менее дорогие объекты.
# Общие выводы:
# Главные факторы, влияющие на стоимость недвижимости, — это площадь (общая и жилая), количество комнат, высота потолков, а также площадь кухни. Эти показатели имеют наибольшие положительные корреляции с ценой.
# Ряд факторов имеют слабую корреляцию с ценой, такие как наличие парков и прудов в радиусе 3000 метров, этажность здания, этаж квартиры и наличие балкона.
# Близость к аэропорту, паркам и прудам, а также открытая планировка и наличие статуса студии оказывают минимальное влияние на стоимость недвижимости.
# Близость к центру города в данной выборке имеет отрицательную корреляцию с ценой, что может свидетельствовать о специфике рынка недвижимости в данном районе или о включении менее дорогих объектов, расположенных в центре.
# Эти выводы могут помочь лучше понять, какие факторы наиболее значимы при оценке стоимости недвижимости в данной выборке.

# #рассчитаем коэффициенты корреляции Пирсона и Спирмена
#
#
#
# # Вычисление коэффициентов корреляции Пирсона
pearson_corr_ceiling_height = df_subset['last_price'].corr(df_subset['ceiling_height'], method='pearson')
pearson_corr_total_area = df_subset['last_price'].corr(df_subset['total_area'], method='pearson')
#
# # Вычисление коэффициентов корреляции Спирмена
spearman_corr_ceiling_height = df_subset['last_price'].corr(df_subset['ceiling_height'], method='spearman')
spearman_corr_total_area = df_subset['last_price'].corr(df_subset['total_area'], method='spearman')
#
# # Печать результатов
print(f"Коэффициент корреляции Пирсона для last_price и ceiling_height: {pearson_corr_ceiling_height:.4f}")
print(f"Коэффициент корреляции Спирмена для last_price и ceiling_height: {spearman_corr_ceiling_height:.4f}")
print(f"\nКоэффициент корреляции Пирсона для last_price и total_area: {pearson_corr_total_area:.4f}")
print(f"Коэффициент корреляции Спирмена для last_price и total_area: {spearman_corr_total_area:.4f}")
#
# Коэффициент корреляции Пирсона для last_price и ceiling_height: 0.3697
# Коэффициент корреляции Спирмена для last_price и ceiling_height: 0.3511
#
# Коэффициент корреляции Пирсона для last_price и total_area: 0.7196
# Коэффициент корреляции Спирмена для last_price и total_area: 0.6946

# На основании предоставленных коэффициентов корреляции Пирсона и Спирмена для различных переменных можно сделать следующие выводы:
#
# 1. Корреляция между last_price и ceiling_height:
# Коэффициент корреляции Пирсона: 0.3697
# Коэффициент корреляции Спирмена: 0.3511
# Выводы:
#
# Коэффициент Пирсона равен 0.3697, что указывает на слабую до умеренной положительную линейную связь между стоимостью недвижимости (last_price) и высотой потолка (ceiling_height). Это означает, что в среднем более высокие потолки могут ассоциироваться с более высокой ценой недвижимости.
# Коэффициент Спирмена равен 0.3511, что также указывает на слабую до умеренной положительную связь, но этот коэффициент учитывает ранжирование данных. То, что коэффициенты Пирсона и Спирмена близки, говорит о том, что связь между last_price и ceiling_height примерно линейна и не слишком зависит от выбросов или нелинейных зависимостей.
# 2. Корреляция между last_price и total_area:
# Коэффициент корреляции Пирсона: 0.7196
# Коэффициент корреляции Спирмена: 0.6946
# Выводы:
#
# Коэффициент Пирсона равен 0.7196, что указывает на сильную положительную линейную связь между стоимостью недвижимости (last_price) и общей площадью (total_area). Это говорит о том, что увеличение площади недвижимости связано с существенным увеличением её стоимости.
# Коэффициент Спирмена равен 0.6946, что также указывает на сильную положительную связь. Близость значений Пирсона и Спирмена указывает на то, что связь между этими переменными в значительной степени линейна и не сильно искажена выбросами или нелинейными отношениями.
# Общие выводы:
# Корреляция с высотой потолка (ceiling_height): Связь между высотой потолков и стоимостью недвижимости существует, но она слабая до умеренной. Это означает, что высота потолка влияет на стоимость, но не является основным фактором.
# Корреляция с общей площадью (total_area): Сильная положительная корреляция показывает, что площадь недвижимости является значительным фактором, влияющим на её стоимость. Чем больше площадь, тем выше цена, и эта зависимость выражена линейно.
# В целом, общая площадь недвижимости имеет гораздо более сильное влияние на её стоимость, чем высота потолков.
# #
# # 6.	Выполнена визуализация данных.
# # + 'ящик с усами' из пункта про обработку аномалий
# #тепловая карта на основе корр. матрицы
sb.heatmap(corr_matrix, annot=True,cmap='coolwarm')#тепловая карта
plt.savefig('Корреляционная матрица.png')
plt.show()
#
#Построим гистограмму для всех числовых столбцов, чтобы визуально
# оценить их распределение.


columns = ['airports_nearest',
    'balcony',
    'ceiling_height',
    'cityCenters_nearest',
    'floor',
    'floors_total',
    'kitchen_area',
    'last_price',
    'living_area',
    'parks_around3000',
    'parks_nearest',
    'ponds_around3000',
    'ponds_nearest',
    'rooms',
    'total_area']
color = ['green'] # списки list
df.hist(column=columns, bins=20, color=color, figsize=(10,8))#bins - колво столбиков figsize размер фигуры
plt.savefig('Частотность параметров.png')
plt.grid(True)
plt.show()








#распределение по стоимости недвижимости

plt.figure(figsize=(10,6))
sb.histplot(df['last_price'], kde = True)
plt.title('Частота распределение данных в зависимости от цены недвижимости')
plt.savefig('Частотность параметров в зависимости от цены.png')
plt.show()



#Создаем scatter plot стоимость недвижимости в зависимости от количества комнат
plt.figure(figsize=(10, 6))
sb.scatterplot(data=df, x='total_area', y='last_price', hue='rooms', palette='viridis', style='rooms')

# Добавляем заголовки и подписи
plt.title('Цена недвижимости в зависимости от общей площади по количеству комнат')
plt.xlabel('Общая площадь (м²)')
plt.ylabel('Стоимость (руб.)')

# Показываем легенду
plt.legend(title='Количество комнат')

# Отображаем график
plt.savefig('Цена недвижимости в зависимости от площади (кол-во комнат).png')
plt.show()

# #Построим scaterplot - зависимости стоймости недвижимости('last_price') по оси у от общей площади  'total_area' по категориям: studio, open_plan , is_apartment (раскрасить по категориям)
# Создаем новый столбец для категории
def determine_category(row):
    if row['studio']:
        return 'studio'
    elif row['open_plan']:
        return 'open_plan'
    elif row['is_apartment']:
        return 'is_apartment'
    return 'unknown'

df['category'] = df.apply(determine_category, axis=1)

# Определяем палитру цветов
palette = {
    'studio': 'red',
    'open_plan': 'blue',
    'is_apartment': 'green',
    'unknown': 'yellow'

}

# Создаем scatter plot Стоимость недвижимости в зависимости от общей площади
plt.figure(figsize=(10, 6))
sb.scatterplot(data=df, x='total_area', y='last_price', hue='category', palette=palette, style='category')

# Добавляем заголовки и подписи
plt.title('Стоимость недвижимости в зависимости от общей площади по категориям')
plt.xlabel('Общая площадь (м²)')
plt.ylabel('Стоимость (руб.)')

# Показываем легенду
plt.legend(title='Категория')

# Отображаем график
plt.savefig('Цена недвижимости в зависимости от площади (по категориям кварртир).png')
plt.show()
print(df.value_counts(['is_apartment','studio','open_plan']))
#

# # Подсчет количества квартир по количеству комнат
# room_counts = df['rooms'].value_counts()
# print(room_counts)
# # plt.figure(figsize=(8, 8))
# # plt.pie(room_counts, labels=room_counts.index, autopct='%1.1f%%', startangle=140)
# # plt.axis('equal')  # Сделать круговой диаграмму круговой формы
# # plt.title('Распределение квартир по количеству комнат')
# # plt.show()



# # # 7.	Составлена гипотеза о данных и выполнена проверка соответствующей гипотезы.
# # Аналитическая Задача: выяснить, какие из признаков оказывают большое влияние на ценообразование квартир.
# #Выдвигаем две гипотезы:
# 1.Стоимость недвижимости зависит от ее общей площади
# 2.Стоимость недвижимости зависит от высоты потолка помещений
# С помощью теста Андерсона-Дарлинга проверим нормальность распределения данных для 'last_price', 'total_area' и 'ceiling_height'
result = st.anderson(df['last_price'])
print(f'T-stat:{result.statistic}')
print(f'Critical-Value: {result.critical_values}')
print(f'Уровень значимости:{result.significance_level}')
for i in range(len(result.critical_values)): #создаем переменную ай в диапазоне result.critical_values
 sl,cv = result.significance_level[i], result.critical_values[i] #sl уровень значимости
 if result.statistic <cv:
  print(f'На уровне значимости{sl}% данные last_price кажутся нормально распределены')
 else:
  print(f'На уровне значимости{sl}% данные last_price не кажутся нормально распределены')
#для 'last_price'
#T-stat:919.2783148949675
# Critical-Value: [0.576 0.656 0.787 0.918 1.092]
# Уровень значимости:[15.  10.   5.   2.5  1. ]
# На уровне значимости15.0% данные не кажутся нормально распределены
# На уровне значимости10.0% данные не кажутся нормально распределены
# На уровне значимости5.0% данные не кажутся нормально распределены
# На уровне значимости2.5% данные не кажутся нормально распределены
# На уровне значимости1.0% данные не кажутся нормально распределены
#для 'total_area'
result = st.anderson(df['total_area'])
print(f'T-stat:{result.statistic}')
print(f'Critical-Value: {result.critical_values}')
print(f'Уровень значимости:{result.significance_level}')
for i in range(len(result.critical_values)): #создаем переменную ай в диапазоне result.critical_values
 sl,cv = result.significance_level[i], result.critical_values[i] #sl уровень значимости
 if result.statistic <cv:
  print(f'На уровне значимости{sl}% данные total_area кажутся нормально распределены')
 else:
  print(f'На уровне значимости{sl}% данные total_area не кажутся нормально распределены')

#для 'ceiling_height'
result = st.anderson(df['ceiling_height'])
print(f'T-stat:{result.statistic}')
print(f'Critical-Value: {result.critical_values}')
print(f'Уровень значимости:{result.significance_level}')
for i in range(len(result.critical_values)): #создаем переменную ай в диапазоне result.critical_values
 sl,cv = result.significance_level[i], result.critical_values[i] #sl уровень значимости
 if result.statistic <cv:
  print(f'На уровне значимости{sl}% данные ceiling_height кажутся нормально распределены')
 else:
  print(f'На уровне значимости{sl}% данные ceiling_height не кажутся нормально распределены')
#Тест Андерсона-Дарлинга был использован для проверки нормальности распределения данных переменной 'last_price', 'total_area' и 'ceiling_height'.
#для 'last_price'
# Основные результаты:
# Статистика теста (T-stat): 919.2783
# Критические значения (Critical-Value): [0.576, 0.656, 0.787, 0.918, 1.092]
# Уровни значимости (Significance Levels): [15%, 10%, 5%, 2.5%, 1%]
# Интерпретация:
# Сравнение статистики теста с критическими значениями:
#
# Статистика теста (919.2783) намного выше критических значений для всех уровней значимости (15%, 10%, 5%, 2.5%, 1%).
# Это указывает на то, что на всех уровнях значимости данные не соответствуют нормальному распределению.
# Заключение по уровням значимости:
#
# На уровне значимости 15%: Данные не кажутся нормально распределены.
# На уровне значимости 10%: Данные не кажутся нормально распределены.
# На уровне значимости 5%: Данные не кажутся нормально распределены.
# На уровне значимости 2.5%: Данные не кажутся нормально распределены.
# На уровне значимости 1%: Данные не кажутся нормально распределены.
# Итог:
# Тест Андерсона-Дарлинга показал, что данные переменной last_price значительно отклоняются от нормального распределения. На всех проверенных уровнях значимости (от 15% до 1%) гипотеза о нормальности распределения была отвергнута. Это указывает на то, что распределение last_price не является нормальным, и для анализа этих данных могут потребоваться методы, которые не предполагают нормальности, или применение методов нормализации данных.


# При ненормальном распределении данных будем использовать U-тест Манна-Уитни. Этот тест проверяет, различаются ли медианы или средние значения двух независимых групп.
# #Выдвигаем две гипотезы:
# 1.Стоимость недвижимости не зависит от ее общей площади
#H₀: Средние значения стоимости недвижимости не различаются между группами с меньшей и большей общей площадью. (Стоимость недвижимости не зависит от общей площади).

#H₁: Средние значения стоимости недвижимости различаются между группами с меньшей и большей общей площадью.
# (Стоимость недвижимости зависит от общей площади).



mean_area = df['total_area'].mean()

group_small_area = df[df['total_area'] <= mean_area]['last_price']
group_large_area = df[df['total_area'] > mean_area]['last_price']

# Применяем U-тест Манна-Уитни для проверки гипотезы
u_stat, p_value = st.mannwhitneyu(group_small_area, group_large_area)

# Выводим результаты
print(f'U-Statistic: {u_stat}')
print(f'p-value: {p_value:.2f}')

# Интерпретируем результат
# Интерпретируем результат
if p_value < 0.05:
    print("Есть статистически значимые различия между стоимостью недвижимости для двух групп (отвергаем нулевую гипотезу H₀).")
else:
    print("Нет статистически значимых различий между стоимостью недвижимости для двух групп (принимаем нулевую гипотезу H₀).")


# Построение гистограммы
plt.figure(figsize=(10, 6))

# Гистограмма для недвижимости с меньшей общей площадью
plt.hist(group_small_area, bins=30, alpha=0.5, color='orange', label='Меньшая общая площадь')

# Гистограмма для недвижимости с большей общей площадью
plt.hist(group_large_area, bins=30, alpha=0.5, color='purple', label='Большая общая площадь')

# Добавляем подписи и легенду
plt.title('Распределение стоимости недвижимости в зависимости от общей площади')
plt.xlabel('Стоимость недвижимости')
plt.ylabel('Количество объектов')
plt.legend()

# Сохраняем график
plt.savefig('Распределение стоимости недвижимости в зависимости от общей площади.png')

# Отображаем график
plt.show()


# U-Statistic: 15956653.5
# p-value: 0.0
#Вывод по 1 гипотезе:
#Есть статистически значимые различия между стоимостью недвижимости для двух групп Нулевая гипотеза(H0) принимается.
#cтоимость зависит от общей площади


# 2.Стоимость недвижимости не зависит от высоты потолка
# Нулевая и альтернативная гипотезы:
# H₀: Средние значения стоимости недвижимости не различаются между группами с меньшей и большей высотой потолков. (Стоимость недвижимости не зависит от высоты потолка)
#H₁: Средние значения стоимости недвижимости различаются между группами с меньшей и большей высотой потолков. (Стоимость недвижимости зависит от высоты потолка)
mean_ceiling_height = df['ceiling_height'].mean()
high_ceiling = df[df['ceiling_height'] > mean_ceiling_height]['last_price']
low_ceiling = df[df['ceiling_height'] <= mean_ceiling_height]['last_price']


# Выполним тест Манна-Уитни
u_statistic, p_value = st.mannwhitneyu(high_ceiling,low_ceiling)

# Выводим результаты
print(f'U-Statistic: {u_statistic}')
print(f'p-value: {p_value:.2f}')  # Округляем p-value до 3 знаков

# Интерпретация результатов
if p_value < 0.05:
    print('Есть статистически значимые различия в стоимости недвижимости между группами с разной высотой потолка (отвергаем нулевую гипотезу H₀).')
else:
    print('Нет статистически значимых различий в стоимости недвижимости между группами с разной высотой потолка (принимаем нулевую гипотезу H₀).')

# Построение гистограммы
plt.figure(figsize=(10, 6))

# Гистограмма для недвижимости с высокими потолками
plt.hist(high_ceiling, bins=30, alpha=0.5, color='blue', label='Высокие потолки')

# Гистограмма для недвижимости с низкими потолками
plt.hist(low_ceiling, bins=30, alpha=0.5, color='green', label='Низкие потолки')

# Добавляем подписи и легенду
plt.title('Распределение стоимости недвижимости в зависимости от высоты потолков')
plt.xlabel('Стоимость недвижимости')
plt.ylabel('Количество объектов')
plt.legend()
plt.savefig('Распределение стоимости недвижимости в зависимости от высоты потолков.png')
# Отображаем график
plt.show()
#Есть статистически значимые различия в стоимости недвижимости между группами с разной высотой потолка.Нулевая гипотеза(H0) принимается.
#cтоимость недвижимости зависит от высоты потолка

#Задача: выяснить, какие из признаков оказывают большое влияние на ценообразование квартир.
# Выводы по проделанной работе

# Описательная статистика после удаления аномалий и очистки датасета

# Вариативность данных: Данные характеризуются значительной вариативностью, что проявляется в высоких значениях стандартного отклонения для многих параметров, таких как стоимость, площадь, высота потолков и расстояние до аэропортов и центров города. Это указывает на широкий диапазон характеристик недвижимости в выборке.
# Наличие выбросов: Большие различия между минимальными, максимальными значениями и перцентилями (например, стоимость и площадь) могут свидетельствовать о наличии выбросов в данных.
# Сбалансированность выборки: Медианы для большинства показателей находятся близко к средним значениям, что говорит о сбалансированности данных и отсутствии значительных перекосов.
# В целом, данные выглядят достаточно полными и репрезентативными, но возможны выбросы, которые следует учитывать при дальнейшем анализе.

# #Корреляционная матрица показала, что наибольшую линейную зависимость c last price
# # 1. Сильные и умеренные положительные корреляции:
# # Общая площадь (total_area): Корреляция с ценой составляет 0.719, что указывает на сильную положительную связь. Это значит, что с увеличением площади квартиры цена также возрастает, что логично, так как большая площадь обычно подразумевает более высокую стоимость.
# # Жилая площадь (living_area): Корреляция 0.567 также указывает на положительную связь, хотя и не такую сильную, как у общей площади. Это свидетельствует о том, что жилая площадь значимо влияет на цену, но меньшей мерой, чем общая площадь.
# # Площадь кухни (kitchen_area): Корреляция 0.469 указывает на умеренную положительную связь. Большая кухня может повышать стоимость, особенно если она соответствует высоким стандартам и удобству.
# # Количество комнат (rooms): Корреляция 0.417 также свидетельствует о положительной связи. Это указывает на то, что квартиры с большим числом комнат, как правило, дороже, хотя эта связь менее сильна, чем с общей площадью.
# # Высота потолков (ceiling_height): Корреляция 0.370 показывает умеренную положительную связь, что подтверждает идею о том, что более высокие потолки могут быть связаны с более высокими ценами, возможно, из-за восприятия большего пространства и роскоши.
# # 2. Слабые положительные корреляции:
# # Количество прудов в радиусе 3000 метров (ponds_around3000): Корреляция 0.281 указывает на слабую положительную связь, что может говорить о том, что наличие прудов вблизи может слегка повышать цену недвижимости.
# # Количество парков в радиусе 3000 метров (parks_around3000): Корреляция 0.265 также указывает на слабую положительную связь, предполагая, что наличие парков в окрестностях также может слегка увеличивать стоимость.
# # Общая этажность здания (floors_total): Корреляция 0.148 является слабой положительной, что может свидетельствовать о незначительном влиянии высоты здания на стоимость недвижимости.
# # Этаж квартиры (floor): Корреляция 0.116 также слабая, что указывает на небольшое влияние этажа на стоимость. Возможно, квартиры на средних этажах могут быть дороже, чем на пер
#Гипотезы
# U-Statistic: 14904903.0
# p-value: 0.00
# Есть статистически значимые различия между стоимостью недвижимости для двух групп (отвергаем нулевую гипотезу H₀).
# U-Statistic: 54915263.5
# p-value: 0.00
# Есть статистически значимые различия в стоимости недвижимости между группами с разной высотой потолка (отвергаем нулевую гипотезу H₀).