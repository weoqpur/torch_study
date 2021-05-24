import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# 데이터 선언
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# 모델 초기화
W = torch.zeros((2, 1), requires_grad=True) # 크기는 2 x 1
b = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = optim.SGD([W, b], lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # x의 크기가 변해도 코드를 수정하지 않아도 됨
    # hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(W) + b)))
    # 위 코드와 동일
    hypothesis = torch.sigmoid(x_train.matmul(W) + b)

    # cost와 cost의 평균 구하기
    # cost = -(y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis)).mean()
    # 위 코드와 동일
    cost = F.binary_cross_entropy(hypothesis, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

# W와 b는 훈련 후의 값을 가지고 있다.
# W와 b의 예측값을 출력해보자.
hypothesis = torch.sigmoid(x_train.matmul(W) + b)
print(hypothesis)

# 0과 1로 분리하기
prediction = hypothesis >= torch.FloatTensor([0.5])
print(prediction)

# W와 b의 값
print(W)
print(b)