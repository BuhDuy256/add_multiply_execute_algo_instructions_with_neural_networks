# Neural Networks - Addition, Multiplication & Permutation Algorithms

Dự án này triển khai và kiểm chứng khả năng học chính xác (exact learning) các thuật toán cộng, nhân và hoán vị sử dụng Neural Tangent Kernels (NTK).

## Yêu cầu

- Python 3.11+
- Conda (khuyến nghị sử dụng Miniconda hoặc Anaconda)

## Cài đặt

### 1. Clone repository

```bash
git clone <repository-url>
cd add_multiply_execute_algo_instructions_with_neural_networks
```

### 2. Tạo môi trường Conda

```bash
conda env create -f environment.yml
conda activate exact_learning
```

## Cấu trúc dự án

```
.
├── training/               # Code huấn luyện (chạy trên Google Colab)
│   ├── original_training.ipynb
│   └── results/
├── validation/             # Code kiểm chứng (chạy local)
│   ├── validate_addition.py
│   ├── validate_multiplication.py
│   ├── validate_permutation.py
│   └── templates/
├── environment.yml         # Cấu hình môi trường Conda
└── README.md
```

---

## Hướng dẫn chạy

### 1. Validation (Chạy trên máy local)

Phần validation dùng để kiểm chứng rằng các thuật toán cộng, nhân và hoán vị được thực thi **chính xác** bởi Neural Tangent Kernel.

#### Chạy kiểm chứng phép cộng:

```bash
cd validation
python validate_addition.py
```

- Kiểm tra tất cả các cặp số với độ dài bit từ 1 đến 10.
- Output: `Addition - bit length: X - all correct`

#### Chạy kiểm chứng phép nhân:

```bash
cd validation
python validate_multiplication.py
```

- Kiểm tra tất cả các cặp số với độ dài bit từ 1 đến 10.
- Output: `Multiplication - Bit length: X - all correct`

#### Chạy kiểm chứng phép hoán vị:

```bash
cd validation
python validate_permutation.py
```

- Kiểm tra với độ dài bit từ 1 đến 4.
- Output: `Permutation(pi=[...]) - bit length: X - all correct`

> **Lưu ý:** Validation chạy trên CPU và sử dụng thư viện `neural_tangents` để tính toán NTK. Không cần GPU.

---

### 2. Training (Chạy trên Google Colab)

> **Quan trọng:** Phần training yêu cầu GPU mạnh (VRAM cao) để huấn luyện mạng neural với hidden dimension lớn (`hidden_dim=50000`). **Khuyến nghị sử dụng Google Colab với GPU T4 hoặc A100.**

#### Các bước thực hiện:

1. **Mở Google Colab:** Truy cập [colab.research.google.com](https://colab.research.google.com)

2. **Upload notebook:**

   - Upload file `training/original_training.ipynb` lên Colab
   - Hoặc: File → Upload notebook

3. **Bật GPU:**

   - Vào **Runtime → Change runtime type**
   - Chọn **Hardware accelerator: GPU** (T4 hoặc A100 nếu có)
   - Nhấn **Save**

4. **Chạy các cell theo thứ tự trong notebook**

#### Nội dung notebook:

Notebook bao gồm 2 thí nghiệm chính:

| Experiment       | k values              | delta | epochs | hidden_dim | Mô tả                       |
| ---------------- | --------------------- | ----- | ------ | ---------- | --------------------------- |
| **Experiment 1** | 2 → 30                | 0.1   | 10000  | 50000      | Đánh giá trên toàn bộ dải k |
| **Experiment 2** | 5, 10, 15, 20, 25, 30 | 0.001 | 10000  | 50000      | Độ chính xác cao hơn        |

#### Tham số cấu hình trong notebook:

| Tham số      | Mô tả                                   |
| ------------ | --------------------------------------- |
| `k`          | Độ dài bit của permutation              |
| `delta`      | Sai số cho phép (1 - accuracy mục tiêu) |
| `epochs`     | Số epochs huấn luyện                    |
| `hidden_dim` | Kích thước hidden layer                 |
| `trials`     | Số lần chạy thử nghiệm                  |
| `batch_size` | Kích thước batch                        |

#### Output:

Kết quả sẽ được lưu vào thư mục `results/` và tự động nén thành file `.zip` để tải về.

---

## Kết quả mong đợi

### Validation

- Tất cả các phép toán cộng, nhân, hoán vị đều cho kết quả **chính xác 100%** khi sử dụng NTK (infinite-width limit).

### Training

- Với `hidden_dim=50000` và `delta=0.001`, mạng neural hữu hạn có thể đạt độ chính xác cao (>99.9%) cho các thuật toán hoán vị.

---

## Tài liệu tham khảo

- Paper gốc về Neural Tangent Kernels và exact learning
- Thư viện [Neural Tangents](https://github.com/google/neural-tangents)