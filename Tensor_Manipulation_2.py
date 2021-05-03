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

# squeeze
ft = torch.FloatTensor([[0], [1], [2]])
print(ft.shape)
print(ft.squeeze()) # element(요소)가 하나인 dimension을 삭제 해줌

# unsqueeze
ft = torch.FloatTensor([0, 1, 2])

print(ft.unsqueeze(0)) # dim = 0에 1을 넣어라
print(ft.unsqueeze(0).shape) # 0번째 dim에 1이 들어감

print(ft.view(1, -1)) # unsqueeze(0)과 똑같은 결과
print(ft.view(1, -1).shape)

print(ft.unsqueeze(1)) # 3x1 의 크기를 가짐

print(ft.unsqueeze(-1)) # 1과 똑같은 결과

# 타입 캐스팅(Type Casting)
lt = torch.LongTensor([1, 2, 3, 4])
print(lt)

print(lt.float()) # float형으로 형변환

bt = torch.ByteTensor([True, False, False, True]) # byte 타입의 텐서
print(bt)
print(bt.long()) # 정수형 텐서
print(bt.float()) # 실수형 텐서

# 연결하기(concatenate)
x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])

print(torch.cat([x, y], dim=0)) # 4x2
print(torch.cat([x, y], dim=1)) # 2x4

# 스택킹(Stacking)
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])

print(torch.stack([x, y, z])) # 3개의 백터를 3x2의 텐서로 만듦

# 스택킹은 많은 연산을 한 번에 축약하고 있다. 위는 아래와 동일하다.

print(torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0))
# x,y,z는 전부 2의 크기를 가졌다. unsqueeze(0)를 하여 3개의 백터를 2차원의 텐서 1x2의 크기로 변경된다.
# 여기에 연결(cat)을 사용하면 3x2의 텐서가 됩니다.

print(torch.stack([x, y, z], dim=1)) # 2x3의 2D 텐서로 만듦

# 0으로 채워진 텐서와 1로 채워진 텐서
x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(x)

print(torch.ones_like(x)) # 입력 텐서와 크기를 동일하게 하면서 값을 0으로 채우기
print(torch.zeros_like(x)) # 위에서 값만 0으로

# 곱
x = torch.FloatTensor([[1, 2], [3, 4]]) # 2x2

print(x.mul(2.)) # 2로 곱하기를 수행한 결과를 출력
print(x) # 저장이 안되어있음

print(x.mul_(2.)) # 2로 곱하기를 수행한 값을 저장하고 출력
print(x) # 저장이 되어있음

