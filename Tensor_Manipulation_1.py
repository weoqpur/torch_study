import numpy as np
import torch

# numpy
t = np.array([0., 1., 2., 3., 4., 5., 6.])

# 차원의 수
print('Rank of t: ', t.ndim)

# 요소의 수
print('shape of t: ', t.shape)

# 2차원 배열
t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])

print('Rank  of t: ', t.ndim)
print('shape of t: ', t.shape)

# View (Reshape)
t = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[6, 7, 8],
               [9, 10, 11]]]) # 3x2x2

ft = torch.FloatTensor(t)
# dimension 별 요소의 수 표시
print(ft.shape)


print(ft.view([-1, 3])) # ? x 3의 크기로 반환
print(ft.view([-1, 3]).shape)

print(ft.view([-1, 1, 3])) # ? x 1 x 3의 크기로 반환
print(ft.view([-1, 1, 3]).shape) # -1, 1 : 요소가 1인 차원을 추 4x1x3

