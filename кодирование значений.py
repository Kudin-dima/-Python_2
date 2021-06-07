#кодирование значений 
import numpy as np
from sklearn import preprocessing

input_labels = ['red','black','red','green','black','yellow','white']

encoder = preprocessing.LabelEncoder() 
# Этот кодировщик берёт столбец с категориальными данными, который был 
# предварительно закодирован в признак, и создаёт для него несколько новых
# столбцов. Числа заменяются на единицы и нули, в зависимости от того, 
# какому столбцу какое значение присуще.
print(encoder.fit(input_labels)) 
# обучающей выборка через метод fit

test_labels = ['green','red','black']
encoded_values = encoder.transform(test_labels) # кодиование списка
print("\nLabels =", test_labels)

print("Encoded values =", list(encoded_values))
# Закодированные значения

#декодирование случайного набора числел
encoded_values = [3,0,4,1]
decoded_list = encoder.inverse_transform(encoded_values) # обратное кодирование
print("\nEncoded values =", encoded_values)

print("Decoded labels =", list(decoded_list))
# декодированные значения