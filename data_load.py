import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

dataset = TensorDataset(x_train, y_train)

# shuffle=True는 모델이 데이터셋의 순서에 익숙해지는 것을 방지하기 위
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = nn.Linear(3, 1) # input_dim=3 output_dim=1
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):

        x_train, y_train = samples
        # H(x) 계산
        prediction = model(x_train)

        # cost 계산
        cost = F.mse_loss(prediction, y_train)

        # cost로 H(x) 계산
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, batch_idx+1, len(dataloader), cost.item()
        ))

new_var = torch.FloatTensor([[73, 80, 75]])
pred_y = model(new_var)

print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y)