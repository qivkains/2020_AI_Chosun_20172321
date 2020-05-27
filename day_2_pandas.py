import pandas as pd

temp = pd.read_csv("data_1.csv")
print (temp)
print(temp['M'])

temp['T'] = 0
print(temp)

# temp['T'][temp['M'] > 3] = 1 주석처리

import matplotlib.pyplot as plt
plt.plot(temp['A'], dashes=[6,2])
plt.show()