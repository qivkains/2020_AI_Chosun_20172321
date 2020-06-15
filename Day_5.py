import pandas as pd
import matplotlib.pyplot as plt

LOCA_1 = pd.read_csv('./DB/12_100010_60.csv')
plt.plot(LOCA_1['ZINST70'])
plt.plot(LOCA_1['QPRZP'])
plt.grid()
plt.show()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

import numpy as np

#print(np.shape(LOCA_1['ZINST70']))

#LOCA_1_VAL_1 =LOCA_1['ZINST70'].to_numpy()
#print(np.shape(LOCA_1_VAL_1), type(LOCA_1_VAL_1))

#LOCA_1_VAL_1 = LOCA_1_VAL_1.reshape((138,1))

#scaler.fit(LOCA_1_VAL_1)
#print(scaler.data_max_)

#LOCA_1_VAL_1_OUT = scaler.transform(LOCA_1_VAL_1)
#plt.plot(LOCA_1_VAL_1_OUT)
#plt.show()

print(LOCA_1.loc[:, ['ZINST70', 'QPRZP']])
LOCA_1_VAL_LIST = LOCA_1.loc[:, ['ZINST70', 'QPRZP']].to_numpy()
print(np.shape(LOCA_1_VAL_LIST), type(LOCA_1_VAL_LIST))
scaler.fit(LOCA_1_VAL_LIST)
LOCA_1_VAL_LIST_OUT = scaler.transform((LOCA_1_VAL_LIST))
plt.plot(LOCA_1_VAL_LIST_OUT)
plt.show()