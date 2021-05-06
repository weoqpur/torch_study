import torch
from torch import optim

# input = output 이라는 결과가 나와야

# Simpler Hypothesis는 bias가 없다
# Dummy Data
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

# 모델 초기화
w = torch.zeros(1)

# learning rate 설정
lr = 0.1

nb_epochs = 10
for epoch in range(nb_epochs + 1):
    # w * x의 곱으로만 계산
    hypothesis = x_train * w

    # 오차를 구함
    cost = torch.mean((hypothesis - y_train) ** 2)
    gradient = torch.sum((w * x_train - y_train) * x_train)

    print('Epoch {:4d}/{} W: {:.3f}, Cost: {:.6f}'.format(epoch, nb_epochs, w.item(), cost.item()))

    # 오차로 H(x) 개선
    w -= lr * gradient
