# x가 여러개인 선형 회귀 문
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# 훈련 데이터
x_train = torch.FloatTensor([[73, 80, 75],
                              [93, 88, 93],
                              [89, 91, 90],
                              [96, 98, 100],
                              [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# 가중치 w와 편향 b 초기화
w = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = optim.SGD([w, b], lr=1e-5)

nb_epochs = 30
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    # hypothesis function : Naive
    # hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b

    # 위와 동일한 식
    # hypothesis function : matrix
    hypothesis = x_train.matmul(w) + b
    # 이렇게 바꿨을 경우 x의 길이가 변해도 코드를 수정할 필요가 없다.

    # cost function : MSE
    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()

    # 기울기 계산
    cost.backward()
    optimizer.step()

    # 로그 출력
    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
        epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()))
