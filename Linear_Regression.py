import numpy as np
import torch
from torch import optim

x_train = torch.FloatTensor([[1], [2], [3]]) # 공부한 시간
y_train = torch.FloatTensor([[2], [4], [6]]) # 맵핑된 점수


# Weight와 bias를 0으 초기화
# 항상 출력을 0으로 예측
w = torch.zeros(1, requires_grad=True) # 학습할 것이라고 명시
b = torch.zeros(1, requires_grad=True)

# 경사 하강법(Gradient Descent)

# 학습시킬 변수들을 리스트로 만들어 넣어주고 lr(learning rate)도 넣어 줌
optimizer = optim.SGD([w, b], lr=0.01)

nb_epochs = 1000 # 반복문을 돌릴 횟수

for epoch in range(1, nb_epochs + 1):
    hypothesis = x_train * w + b  # 선형 회귀의 가설(직선의 방정식)

    # 평균 제곱 오차(Mean Squared error)
    cost = torch.mean((hypothesis - y_train) ** 2)  # 오차들을 제곱해 양수로 만든 뒤 평균을 구함

    # 이 세줄은 옵디마이저랑 꼭 붙어다님
    optimizer.zero_grad() # gradient 초기화
    cost.backward() # gradient 계산
    optimizer.step() # 계산된 방향으로 w,b를 개선

    print(cost)

