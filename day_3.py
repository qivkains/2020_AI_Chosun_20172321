import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 데이터 받아오기.
mnist = tf.keras.datasets.mnist
mnist_data = mnist.load_data()

# 데이터 모양 보기.
print(np.shape(mnist_data))
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(np.shape(x_train))
print(x_test[0])

#이미지화
plt.imshow(x_train[0])
plt.title(y_train[0])
plt.show()

#레이어 설계
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(5, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='tanh')
])

out = model.predict(x_train[0:2])
print(out)
print (np.shape(out))

# ---------------------
print(y_train[0], type(y_train[0]))
print(x_train[0], type(x_train[0]))

# make one hot

# print (max(y_train))
temp_y = []
for one_y_val in y_train:
    zero_array = np.zeros(10) # numpy 0으로 채워진 10개의 1차 행렬
    zero_array[one_y_val] = 1
    temp_y.append(zero_array) # 데이터 쌓음
temp_y = np.array(temp_y)
print(type(temp_y))
