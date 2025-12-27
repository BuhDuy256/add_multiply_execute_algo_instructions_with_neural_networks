import torch
import torch.nn as nn

class NTKMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, sigma_w=1.0):
        super(NTKMLP, self).__init__()
        # Paper sử dụng mạng 2 lớp không bias 
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.relu = nn.ReLU()
        self.sigma_w = sigma_w
        self._initialize_weights()

    def _initialize_weights(self):
        # Khởi tạo trọng số theo công thức NTK: std = sigma / sqrt(fan_in) 
        nn.init.normal_(self.fc1.weight, mean=0, std=self.sigma_w / (self.fc1.in_features ** 0.5))
        nn.init.normal_(self.fc2.weight, mean=0, std=self.sigma_w / (self.fc2.in_features ** 0.5))

    def forward(self, x):
        # F(x) = W2 * ReLU(W1 * x) [cite: 117]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x