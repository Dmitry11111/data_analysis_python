# Анализ факторов, влияющих на ценообразование недвижимости
============
--В итоговой работе проведен исследовательский анализ датасета.
## Цель: Проанализировать какие из представленных признаков оказывают наибольшее влияние на ценообразование недвижимости.

### Задачи:  
➜Выполнена предварительная обработка данных.  
➜Выполнен исследовательский анализ данных.  
➜Составлены гипотезы о данных и выполнена проверка соответствующих гипотез.  
➜Подготовлен и опубликован дашборд с выводами по поставленной бизнес-задаче.  

Исходные данные:  
Данные df.info() датасета home price  
<img width="260" alt="print(df info_before" src="https://github.com/user-attachments/assets/f971f40e-ad81-495c-9be8-8f02fe0872f5"> <br>
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

![boxplots_before_after](https://github.com/user-attachments/assets/c1c0ae3d-12d8-48e8-92ef-658bd74b60e1)
