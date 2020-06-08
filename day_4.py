import tensorflow as tf
import numpy as np

# 데이터 받아오기.
mnist = tf.keras.datasets.mnist
mnist_data = mnist.load_data()

# 데이터 모양 보기.
print(np.shape(mnist_data))
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(np.shape(x_train))
print(x_test[0])

# 뉴럴네트워크
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax') # softmax: 최대값 1
])

#
model.compile(optimizer='adam', # adam:알고리즘
                loss='sparse_categorical_crossentropy', # loss: 손실함수(실제 값과 예측값의 차이)
                metrics=['accuracy']) # 정확도
#
print(np.shape(x_train), np.shape(y_train))
print(type(x_train), type(y_train))
model.fit(x_train, y_train, epochs=50) # model.fit: 실제 값과 비교, epochs: 반복 실행 횟수

# 검증
out_net = model.predict(x_test[0:3])
for x_out, y_out in zip(out_net, y_test[0:3]):
    print(np.argmax(x_out), y_out)

# 저장
model.save_weights('save_model')