import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import glob

train_x, train_y = [], []
PARA = ['UHOLEG1', 'UHOLEG2', 'UHOLEG3', 'ZINST58']
for one_file in glob.glob('DB/*.csv'):
    LOCA = pd.read_csv(one_file)
    if len(train_x) == 0:
        train_x = LOCA.loc[:, PARA].to_numpy()
        train_y = LOCA.loc[:, ['Normal_0']].to_numpy()
    else:
        get_x = LOCA.loc[:, PARA].to_numpy()
        get_y = LOCA.loc[:, ['Normal_0']].to_numpy()
        train_x = np.vstack((train_x, get_x))
        train_y = np.vstack((train_y, get_y))
    print(f'SHAPE : {np.shape(train_x)},|'
          f'SHAPE : {np.shape(train_y)}')
print('DONE')


#plt.plot(train_x[0:60])
#plt.plot(train_y[0:60])
#plt.show()

import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(np.shape(train_x)[1]),
    tf.keras.layers.Dense(5000),
    tf.keras.layers.Dense(5000),
    tf.keras.layers.Dense(5000),
    tf.keras.layers.Dense(np.shape(train_y)[1]+1,
                          activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_x, train_y, epochs=1000)

out_trained = model.predict(train_x[0:60])

#plt.plot(train_x)
plt.plot(train_y[0:60], label='train_y')
#plt.plot(out_max)
#plt.plot(out)
plt.plot(out_trained, label='Out')
plt.legend()
plt.show()