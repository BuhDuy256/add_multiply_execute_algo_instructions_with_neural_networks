import torch
from torch.utils.data import Dataset

class PermutationDataset(Dataset):
    def __init__(self, P: torch.Tensor):
        super().__init__()
        self.P = P
        self.n = P.shape[0]
        # Huấn luyện trên các vector cơ sở trực giao [cite: 280]
        self.basis = torch.eye(self.n)

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        x = self.basis[idx]
        y = self.P @ x
        return x, y

def generate_permutation(n):
    # Tạo ma trận hoán vị ngẫu nhiên
    P = torch.eye(n)[torch.randperm(n)]
    return P

def generate_test_set(k, num_samples, nnz=2):
    # Tạo test set với nnz bit 1 (mặc định là 2 như trong Exp 2) [cite: 488]
    x_test = torch.zeros((num_samples, k))
    for i in range(num_samples):
        indices = torch.randperm(k)[:nnz]
        x_test[i, indices] = 1.0
    
    # Chuẩn hóa đầu vào theo sqrt(nnz) 
    x_test_norm = x_test / (nnz ** 0.5)
    return x_test_norm, x_test # Trả về cả bản chuẩn hóa và bản bit gốc
