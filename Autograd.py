import torch

# 2w**2 + 5
w = torch.tensor(2.0, requires_grad=True)

y = w**2
z = 2*y + 5

# 기울기 계산
z.backward()

print('수식을 w로 미분한 값 : {}'.format(w.grad))