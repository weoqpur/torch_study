import torch
import torch.nn as nn

# 배치 크기 * 채널 * 높이(height) * 너비(widht)의 크기의 텐서를 선언
inputs = torch.Tensor(1, 1, 28, 28)
print('텐서의 크기 : {}'.format(inputs.shape))

# 1채널 짜리를 입력받아서 32채널을 뽑아내는데 커널 사이즈는 3이고 패딩은 1입니다.
conv1 = nn.Conv2d(1, 32, 3, padding=1) # 합성곱층
print(conv1)

# 32채널 짜리를 입력받아서 64채널을 뽑아내는데 커널 사이즈는 3이고 패딩은 1입니다.
conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 합성곱층
print(conv2)

# 정수 인자 하나만 넣는다 커널 사이즈와 스트라이드가 둘 다 해당값으로 지정
pool = nn.MaxPool2d(2) # 풀링층
print(pool)

out = conv1(inputs)
# out_channel을 32로 지정하여 32로 나왔고 패딩을 1폭으로 하고 3x3 커널을 사용하여 크기가 보존됨
print(out.shape) # 1, 32, 28, 28

out = pool(out) # 풀링층 통과
print(out.shape)

out = conv2(out)
print(out.shape)

out = pool(out)
print(out.shape)

# 첫번째 차원인 배치 차원은 그대로 두고 나머지는 펼쳐라
out = out.view(out.size(0), -1)
print(out.shape)

# 전결합층(Fully-Connteced layer)

fc = nn.Linear(3136, 10) # input_dim = 3136, output_dim = 10
out = fc(out)
print(out.shape)