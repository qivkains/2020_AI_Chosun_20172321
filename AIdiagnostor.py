import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import glob
import pickle

train_x, train_y = [], []
PARA = ['ZINST101', 'ZINST102', 'WSTM1', 'WSTM2', 'WSTM3', 'ZINST78',
        'ZINST77', 'ZINST76', 'ZINST72', 'ZINST71', 'ZINST70', 'ZINST75',
        'ZINST74', 'ZINST73', 'ZINST58', 'ZINST56', 'UPRT', 'ZINST63', 'KBCDO15',
        'ZINST26', 'ZINST22', 'UCTMT', 'WFWLN1', 'WFWLN2', 'WFWLN3', 'BPORV',
        'KLAMPO147', 'KLAMPO148', 'KLAMPO149', 'ZINST36', 'KLAMPO9', 'KFAST', 'KLAMPO70']

for one_file in glob.glob('DB/*.csv'):
    LOCA = pd.read_csv(one_file)
    if not 'Accident_nub' in LOCA.keys():
        print('T')
    for keys in PARA:
        if not keys in LOCA.keys():
            print(keys)

    if len(train_x) == 0:
        train_x = LOCA.loc[:, PARA].to_numpy()
        train_y = LOCA.loc[:, ['Accident_nub']].to_numpy()
    else:
        get_x = LOCA.loc[:, PARA].to_numpy()
        get_y = LOCA.loc[:, ['Accident_nub']].to_numpy()
        train_x = np.vstack((train_x, get_x))
        train_y = np.vstack((train_y, get_y))
    print(f'SHAPE : {np.shape(train_x)},|'
          f'SHAPE : {np.shape(train_y)}')

# with open('DB.pkl', 'wb') as f:
    # pickle.dump([train_x, train_y], f)

scaler = MinMaxScaler()
scaler.fit(train_x)

train_x = scaler.transform(train_x)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


print(np.shape(train_x), np.shape(train_y))
print('DONE')

import tensorflow as tf



inputs = tf.keras.Input(len(PARA))
hiden1 = tf.keras.layers.Dense(300, activation='relu')(inputs)
hiden2 = tf.keras.layers.Dense(150, activation='relu')(hiden1)
output = tf.keras.layers.Dense(5, activation='softmax')(hiden2)
model = tf.keras.models.Model(inputs, output)

model.load_weights('201723211.h5')

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_x, train_y, epochs=111)

out_trained = model.predict(train_x[0:60])
print(np.shape(out_trained))

model.save_weights('20172321.h5')

plt.plot(train_y[0:60], label='Train_Y')

temp = {0:[], 1:[], 2:[], 3:[], 4:[]}
for _ in out_trained:
    for __ in range(5):
        temp[__].append(_[__])

plt.plot(temp[0], label='0')
plt.plot(temp[1], label='1')
plt.plot(temp[2], label='2')
plt.plot(temp[3], label='3')
plt.plot(temp[4], label='4')
plt.legend()
plt.show()