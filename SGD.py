import torch
from torch import optim

# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# 모델 초기화
w = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# requires_grad=True로 되어있으면 자동 미분기능이 적용 됨
# requires_grad=True로 되어있는 텐서를 연산 할 경우 계산 그래프가 생성되며
# backward 함수를 호출하면 그래프로 부터 자동으로 미분이 계산 됨

# optimizer 설정
optimizer = optim.SGD([w, b], lr=0.01)

# 원하는 만큼 반복
nb_epochs = 2000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = x_train * w + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    # pytorch는 미분을 통해 얻은 기울기를 이전에 계산된 기울기 값에 누적을 시킨다
    optimizer.zero_grad()
    # optimizer.zero_grad()를 통해 미분값을 계속 0으로 초기화 시켜준다

    cost.backward()
    optimizer.step()

    # 100번 마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, w.item(), b.item(), cost.item()
        ))