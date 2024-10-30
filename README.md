# Анализ факторов, влияющих на ценообразование недвижимости
============
## В итоговой работе проведен исследовательский анализ датасета.
## Цель: Проанализировать какие из представленных признаков оказывают наибольшее влияние на ценообразование недвижимости.

### Задачи:  
➜Выполнена предварительная обработка данных.  
➜Выполнен исследовательский анализ данных.  
➜Составлены гипотезы о данных и выполнена проверка соответствующих гипотез.  
➜Подготовлен и опубликован дашборд с выводами по поставленной бизнес-задаче.  

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
### 1. Стоимость недвижимости (last_price):  
#Средняя стоимость составляет 5.292 миллиона рублей, с медианой в 4.5 миллиона рублей.
Стандартное отклонение достаточно велико (3.22 миллиона рублей), что указывает на
значительную вариативность цен.
#Минимальная стоимость — 1 миллион рублей, максимальная — 36 миллионов рублей, что
подтверждает наличие объектов как низкого, так и высокого ценового диапазона.
#25-й и 75-й перцентили (3.4 и 6.2 миллиона рублей) показывают, что большая часть объектов
находится в диапазоне от 3.4 до 6.2 миллионов рублей.
### 2. Общая площадь (total_area):  
#Средняя площадь составляет 55 кв.м., с медианой 50.7 кв.м.
#Стандартное отклонение в 20 кв.м. указывает на значительное расхождение в размерах
квартир.
#Минимальная площадь — 25.36 кв.м., максимальная — 197 кв.м.
#25-й и 75-й перцентили (40 и 65.5 кв.м.) показывают, что большинство квартир имеют
площадь от 40 до 65.5 кв.м.
### 3. Высота потолков (ceiling_height):
# Средняя высота потолков — 2.68 м, с медианой 2.65 м.
# Стандартное отклонение — 0.185 м, что указывает на небольшую вариативность, но с
максимальной высотой до 4.8 м, что может свидетельствовать о наличии редких квартир с
высокими потолками.
### 4. Вариативность данных:   
Данные характеризуются значительной вариативностью, что проявляется в высоких значениях стандартного
отклонения для многих параметров, таких как стоимость, площадь,
высота потолков и расстояние до аэропортов и центров города. Это
указывает на широкий диапазон характеристик недвижимости в
выборке.
### 5. Наличие выбросов:  
Большие различия между минимальными, максимальными значениями и перцентилями (например, стоимость и
площадь) могут свидетельствовать о наличии выбросов в данных.
### 6. Сбалансированность выборки:   
Медианы для большинства показателей находятся близко к средним значениям, что говорит о
сбалансированности данных и отсутствии значительных перекосов.
### В целом, данные выглядят достаточно полными и репрезентативными, но возможны выбросы, которые следует учитывать при дальнейшем анализе.






















