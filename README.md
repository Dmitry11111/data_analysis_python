# Анализ факторов, влияющих на ценообразование недвижимости
============
## В итоговой работе проведен исследовательский анализ датасета.
## Цель: Проанализировать какие из представленных признаков оказывают наибольшее влияние на ценообразование недвижимости.

### Задачи:  
➜Выполнена предварительная обработка данных.  
➜Выполнен исследовательский анализ данных.  
➜Составлены гипотезы о данных и выполнена проверка соответствующих гипотез.  
➜Подготовлен и опубликован дашборд с выводами по поставленной бизнес-задаче.  

### Исходный фаил  [home price](https://example.com/).

Исходные данные:  
Данные df.info() датасета home price  
<img width="260" alt="info_before" src="https://github.com/user-attachments/assets/f971f40e-ad81-495c-9be8-8f02fe0872f5"> <br>    
Данные df.describe() датасета home price  
<img width="493" alt="describe before" src="https://github.com/user-attachments/assets/c8c16f09-032e-4b07-9a9e-c01921c671d3">        

## Предобработка данных
➜Данные очищены от дубликатов, пропущенных значений и аномалий


Пример кода на python:  
```python
##для столбца 'ceiling_height' (высота потолков) мы заменим пропущенные значения на медиану
##посчитаем медиану
median_ceiling_height = float(df_cleaned['ceiling_height'].median())
print("Медиана высоты потолков:", median_ceiling_height)
  
##Заменяем значения высоты потолков меньше 2,45 м на медиану
df_cleaned['ceiling_height'] = np.where(df_cleaned['ceiling_height'] < 2.45, median_ceiling_height,
df_cleaned['ceiling_height'])

#создаем фильтр для удаления аномалий,
filt = ((df_cleaned.last_price.between(np.nanpercentile(df_cleaned.last_price, 1),
np.nanpercentile(df_cleaned.last_price, 99))) #для last price исходя из графика,
задаем нижнюю границу 1 процентиль и 99 процентиль
& (df_cleaned.total_area.between(np.nanpercentile(df_cleaned.total_area, 1),
np.nanpercentile(df_cleaned.total_area, 99))) #для total_area задаем от 1 до 99
процентиля
```
### После удаления аномальных значений осталось 84.33% данных.
### 23699->19912

### Ящичная диаграмма с усами ‘boxplot’(до и после)

![boxplots_before_after](https://github.com/user-attachments/assets/8c514290-a0bc-4b3e-8ed4-842419ac9dd3)  

## Математическая статистика  

#### Данные df.describe() после обработки  

<img width="493" alt="describe before" src="https://github.com/user-attachments/assets/41bf469b-8fdf-4981-be63-c138094d146b">  

➜Основные выводы:  
##### 1. Стоимость недвижимости (last_price):  
#Средняя стоимость составляет 5.292 миллиона рублей, с медианой в 4.5 миллиона рублей.
Стандартное отклонение достаточно велико (3.22 миллиона рублей), что указывает на
значительную вариативность цен.
#Минимальная стоимость — 1 миллион рублей, максимальная — 36 миллионов рублей, что
подтверждает наличие объектов как низкого, так и высокого ценового диапазона.
#25-й и 75-й перцентили (3.4 и 6.2 миллиона рублей) показывают, что большая часть объектов
находится в диапазоне от 3.4 до 6.2 миллионов рублей.
##### 2. Общая площадь (total_area):  
#Средняя площадь составляет 55 кв.м., с медианой 50.7 кв.м.
#Стандартное отклонение в 20 кв.м. указывает на значительное расхождение в размерах
квартир.
#Минимальная площадь — 25.36 кв.м., максимальная — 197 кв.м.
#25-й и 75-й перцентили (40 и 65.5 кв.м.) показывают, что большинство квартир имеют
площадь от 40 до 65.5 кв.м.
##### 3. Высота потолков (ceiling_height):
#Средняя высота потолков — 2.68 м, с медианой 2.65 м.
#Стандартное отклонение — 0.185 м, что указывает на небольшую вариативность, но с
максимальной высотой до 4.8 м, что может свидетельствовать о наличии редких квартир с
высокими потолками.
##### 4. Вариативность данных:   
Данные характеризуются значительной вариативностью, что проявляется в высоких значениях стандартного
отклонения для многих параметров, таких как стоимость, площадь,
высота потолков и расстояние до аэропортов и центров города. Это
указывает на широкий диапазон характеристик недвижимости в
выборке.
##### 5. Наличие выбросов:  
Большие различия между минимальными, максимальными значениями и перцентилями (например, стоимость и
площадь) могут свидетельствовать о наличии выбросов в данных.
##### 6. Сбалансированность выборки:   
Медианы для большинства показателей находятся близко к средним значениям, что говорит о
сбалансированности данных и отсутствии значительных перекосов.
##### В целом, данные выглядят достаточно полными и репрезентативными, но возможны выбросы, которые следует учитывать при дальнейшем анализе.

## Исследовательский анализ данных:  
### ➜Поиск закономерностей  
#Корреляции с last_price:
#last_price 1.000000
#total_area 0.719583
#living_area 0.566810
#kitchen_area 0.468916
#rooms 0.417236
#ceiling_height 0.369747
#ponds_around3000 0.280909

#Коэффициент корреляции Пирсона для last_price и ceiling_height: 0.3697  
#Коэффициент корреляции Спирмена для last_price и ceiling_height: 0.3511  
#Коэффициент корреляции Пирсона для last_price и total_area: 0.7196  
#Коэффициент корреляции Спирмена для last_price и total_area: 0.6946  

Тест Андерсона-Дарлинга был использован для проверки нормальности  
распределения данных переменных -'total_area' и 'ceiling_height' 'total_area’:  
T-stat:398.936978011101  
Critical-Value: [0.576 0.656 0.787 0.918 1.092]  
Уровень значимости:[15. 10. 5. 2.5 1. ] -на всех уровнях значимости данные не
соответствуют нормальному распределению.  
'ceiling_height’:  
T-stat:1738.0876839728517  
Critical-Value: [0.576 0.656 0.787 0.918 1.092]  
Уровень значимости:[15. 10. 5. 2.5 1. ] -на всех уровнях значимости данные не
соответствуют нормальному распределению.  
Коэффициент Пирсона равен 0.3697, что указывает на слабую до умеренной положительную линейную связь  
между стоимостью недвижимости (last_price) и высотой потолка (ceiling_height)  

## Исследовательский анализ данных:  
### ➜Визуализация  

![Кореляционная матрица](https://github.com/user-attachments/assets/ef80e733-5072-46c9-8224-745d125950ff)  
![Цена недвижимости в зависимости от площади (кол-во комнат)](https://github.com/user-attachments/assets/42901c9f-ebc0-4e9a-ae69-8eab79b46236)  
![Распределение по стоимости](https://github.com/user-attachments/assets/cef2317d-a460-4892-b595-4d8fa7a13094)  

## Исследовательский анализ данных: 
### Были сформулированы две гипотезы:  

#### 1.Нулевая гипотеза(H0). Стоимость недвижимости(last_price) не зависит от ее общей площади(total area). (средние значения стоимости недвижимости не различаются между группами с меньшей и большей общей площадью).  
#### 2.Нулевая гипотеза(H0). Стоимость недвижимости(last_price) не зависит от высоты потолка помещений(ceiling_height). (средние значения стоимости недвижимости не различаются между группами с меньшей и большей высотой потолков).  
### ➜Получение результатов и их интерпретация  
U-тест Манна-Уитни показал:
U-Statistic: 14904903.0
p-value: 0.00
Есть статистически значимые различия между стоимостью недвижимости для двух групп с
меньшей и большей общей площадью (отвергаем нулевую гипотезу H₀).
U-Statistic: 54915263.5
p-value: 0.00
Есть статистически значимые различия в стоимости недвижимости между группами с разной
высотой потолка (отвергаем нулевую гипотезу H₀).

## Исследовательский анализ данных:  
### ➜Визуализация гипотез    
![Распределение стоимости недвижимости в зависимости от общей площади](https://github.com/user-attachments/assets/17505c49-5c14-4e28-b968-11aa235d6226)  
Две группы значений с меньшей и большей общей площадью  
![Распределение стоимости недвижимости в зависимости от высоты потолков](https://github.com/user-attachments/assets/ba87528b-c9e6-453e-8306-553d3d6f95c6)  
Две группы значений с меньшей и большей высотой потолка  

## Результаты и выводы:

### 1. Исследование влияния параметров на стоимость недвижимости  
Из корреляционной матрицы были найдены признаки, которые влияют на ценообразование.
#### 1.) Сильные и умеренные положительные корреляции:    
-Общая площадь (total_area): Корреляция с ценой составляет 0.719, что указывает на сильную положительную связь.
-Жилая площадь (living_area): Корреляция 0.567 также указывает на положительную связь, хотя и не такую сильную, как у общей площади.
-Площадь кухни (kitchen_area): Корреляция 0.469 указывает на умеренную положительную связь.
-Количество комнат (rooms): Корреляция 0.417 также свидетельствует о положительной связи.
-Высота потолков (ceiling_height): Корреляция 0.370 показывает умеренную положительную связь, что подтверждает идею о том,
что более высокие потолки могут быть связаны с более высокими ценами
#### 2.) Слабые положительные корреляции:  
-Количество прудов в радиусе 3000 метров (ponds_around3000): Корреляция 0.281 указывает на слабую положительную связь
-Количество парков в радиусе 3000 метров (parks_around3000): Корреляция 0.265 также указывает на слабую положительную связь,
-Общая этажность здания (floors_total): Корреляция 0.148 является слабой положительной
-Этаж квартиры (floor): Корреляция 0.116 также слабая, что указывает на небольшое влияние этажа на стоимость.
### 2. Были выбраны 2 параметра (‘totat_area’ и ceiling_height) и составлены соответствующие гипотезы.  
#### Вывод по первой гипотезе:  
Есть статистически значимые различия между стоимостью недвижимости для двух групп с меньшей
и большей площадью (отвергаем нулевую гипотезу H₀ и принимаем альтернативную).
Стоимость недвижимости(last_price) зависит от ее общей площади(total area).
#### Вывод по второй гипотезе:  
Есть статистически значимые различия в стоимости недвижимости между группами с разной высотой потолка
(отвергаем нулевую гипотезу H₀ и принимаем альтернативную). Стоимость недвижимости(last_price) зависит от высоты потолка
помещений(ceiling_height).

##Дашборд[➜Опубликованный дашборд](https://datalens.yandex/xvuy1a83xevsk).



