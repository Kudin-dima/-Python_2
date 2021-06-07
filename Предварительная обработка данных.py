  #Предварительная обработка данных
import numpy as np
from sklearn import preprocessing

input_data = np.array([[2.1, -1.9, 5.5],
                      [-1.5, 2.4, 3.5], 
                      [0.5, -7.9, 5.6], 
                      [5.9, 2.3, -5.8]])

#Применение методов предварительной обработки

#Бинаризация 
data_binarized = preprocessing.Binarizer(threshold = 0.5).transform(input_data)
print("\tБинаризация\n", data_binarized)

#Среднее удаление
print("\n\tСреднее удаление\nmean = ", input_data.mean(axis = 0)) #вычисляет среднее значение по сплющенному массиву, в котором axis=0 вдоль строк
print("Standart deviation = ", input_data.std(axis = 0)) #вычисляет среднеквадратичное (стандартное) отклонение

data_scaled = preprocessing.scale(input_data) #Метод preprocessing. scale() полезен при стандартизации точек данных. Разделить все точки данных на среднее значение и вычесть стандартное отклонение для каждой точки данных.
print("\nmean = ", data_scaled.mean(axis = 0))
print("standart deviation = ", data_scaled.std(axis = 0))

#Пересчет 
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0,1)) #MinMaxScaler. Для каждого значения в объекте MinMaxScaler вычитает минимальное значение в объекте и затем делит на диапазон
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data) #fit_transform означает выполнить некоторые вычисления, а затем выполнить преобразование
print ("\n\tПересчет\n", data_scaled_minmax)

#L1 нормализация
data_normalized_l1 = preprocessing.normalize(input_data, norm = 'l1')#Норма, используемая для нормализации каждой ненулевой выборки (или каждого ненулевого признака, если ось равна 0).
print("\n\tL1 нормализация\n", data_normalized_l1)

#L2 нормализация
data_normalized_l2 = preprocessing.normalize(input_data, norm = 'l2')
print("\n\tL2 нормализация\n", data_normalized_l2)