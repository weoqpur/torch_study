import numpy as np
import torch

# pytorch
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])

print(t.dim()) # 차원의 수
print(t.shape) # 요소의 수
print(t.size()) # 배열의 길이
print(t[0], t[1], t[-1]) # 인덱스로 찾을 수 있음
print(t[2:5], t[4:-1]) # slicing

# 2D array

t = torch.FloatTensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])

print(t.dim()) # 차원의 수
print(t.size()) # shape
print(t[:, 1])
print(t[:, 1].size())
print(t[:, :-1])

# 텐서가 텐서와 연산을 할 경우 서로 크기가 같아야 한다
# 서로 크기가 다른 텐서끼리 연산을 할 경우 torch가 자동으로 크기를 맞춰준다.

# same shape
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print(m1 + m2)

# vactor + scalar
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([3]) # 3 -> [[3, 3]]
print(m1 + m2)

# 2 x 1 vector + 1 x 2 vector
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3], [4]])
print(m1 + m2) # 2 x 2 print

# matmul 최소 크기에 맞춰 연산
m1 = torch.FloatTensor([[1, 2], [3, 4]]) # 세로의 평
m2 = torch.FloatTensor([[1], [2]])
print(m1.matmul(m2)) # 2 x 1

# mul
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print(m1 * m2) # 2 x 2
print(m1.mul(m2)) # m1 * m2

# Mean

# 요소들의 평균
t = torch.FloatTensor([1, 2])
print(t.mean())

# long tensor는 연산을 못 함
t = torch.LongTensor([1, 2])
try:
    print(t.mean())
except Exception as exc:
    print(exc)

# 차원 별로 평균 연산 가능
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)

# dim = 1 : 2번째 차원
print(t.mean())
print(t.mean(dim=0)) # 1 + 3 / 2 = 2 , 2 + 4 / 2 = 3
print(t.mean(dim=1))
print(t.mean(dim=-1))

# sum
print(t.sum())
print(t.sum(dim=0)) # 세로로 합
print(t.sum(dim=1))
print(t.sum(dim=-1))

# max
print(t.max()) # 요소들 중 가장 큰 값 1개
print(t.max(dim=0)) # 세로의 가장 큰 값 1개씩 , 그 요소의 인덱스
print('Max: ', t.max(dim=0) [0])
print('Argmax: ', t.max(dim=0) [1])

print(t.max(dim=1))
print(t.max(dim=-1))
