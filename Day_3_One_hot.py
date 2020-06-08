import numpy as np
input_val = 1

zero = np.zeros(shape=5)
print(zero, type(zero))

zero[input_val] = 1 # 행렬의 1번 칸 값을 1로 변경
print(zero, type(zero))