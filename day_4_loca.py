import pandas as pd
import matplotlib.pyplot as plt
db = pd.read_csv('./DB/loop1 coldleg_loca_20.csv')
print(db)

print(db['BFV122']) # 변수 BFV122 열만 호출
plt.plot(db['BFV122'])
plt.show()