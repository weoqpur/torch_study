import torch

torch.manual_seed(3)
print("랜덤 시드가 3일 때")
for i in range(1, 3):
    print(torch.rand(1))

torch.manual_seed(5)
print("랜덤 시드가 5일 때")
for i in range(1, 3):
    print(torch.rand(1))

torch.manual_seed(3)
print("랜덤 시드가 다시 3일 때")
for i in range(1, 3):
    print(torch.rand(1))

# torch.manual_seed()를 사용하는 이유
# 난수의 발생 순서와 값이 동일하기 때문