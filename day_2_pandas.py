import pandas as pd

temp = pd.read_csv("data_1.csv")
print (temp)
print(temp['M'])

temp['T'] = 0
print(temp)

# temp['T'][temp['M'] > 3] = 1

import matplotlib.pyplot as plt # 매트플롯라이브러리 plt로 명명
plt.plot(temp['A'], dashes=[6,2]) # 매트플롯라이브러리 호출, temp의 A 줄 읽기, 점선 색 6칸 공백 2칸 반복
plt.xlim((0, 5)) # x축 범위 설정
plt.ylim((0, 10)) # y축 범위 설정
plt.show() # 그래프 표시