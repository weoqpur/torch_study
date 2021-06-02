import torch
import torch.nn.functional as F

torch.manual_seed(1)


z = torch.rand(3, 5, requires_grad=True) # 랜덤으로 3x5 텐서를 생

hypothesis = F.softmax(z, dim=1) # 각 샘플에 대해서 소프트맥수 함수를 적용시켜야 하므로 dim=1을 준다.

y = torch.randint(5, (3,)).long() # 5개의 백터에 3개의 원 핫 인코딩을 해줌

y_one_hot = torch.zeros_like(hypothesis)
y_one_hot.scatter_(1, y.unsqueeze(1), 1) #dim=1, index=[[0],[2],[1]]

cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
print(cost)

# Low level
# print(torch.log(F.softmax(z, dim=1)))
# 위와 동일
# High level
print(F.log_softmax(z, dim=1))

# low level cost function
print((y_one_hot * -torch.log(F.softmax(z, dim=1))).sum(dim=1).mean())
print((y_one_hot * - F.log_softmax(z, dim=1)).sum(dim=1).mean())

# high level cost function
print(F.nll_loss(F.log_softmax(z, dim=1), y)) # 원-핫 백터 필요없이 실제 값을 인자로 사용함
print(F.cross_entropy(z, y)) # F.cross_entropy는 F.log_softmax와 F.nll_loss를 포함한다.
