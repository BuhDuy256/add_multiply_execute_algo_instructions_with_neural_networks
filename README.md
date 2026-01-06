
# add_multiply_execute_algo_instructions_with_neural_networks

Repo này hiện mới ở mức **validation / proof-of-concept** (NTK + template matching), chưa phải code training đầy đủ để reproduce toàn bộ phần thực nghiệm của paper.

## Trạng thái hiện tại (so với paper)

- `training/dataset.py`, `training/model.py`, `training/experiments.py` đang **trống** ⇒ chưa có pipeline train/eval để reproduce các bảng/biểu đồ thực nghiệm trong paper.
- `validation/validate_addition.py` và `validation/validate_permutation.py` là các script kiểm chứng logic theo hướng **Neural Tangents / NTK**.
- `validation/validate_multiplication.py` và `validation/templates/multiplication.py` đang **trống**.

Nói cách khác: repo có thể dùng để kiểm tra “đúng logic” cho một phần nhỏ (addition/permutation), nhưng **không đủ để reproduce kết quả thực nghiệm chính**.

## Chạy validation trên laptop

Khuyến nghị chạy theo module để import tương đối hoạt động đúng:

```bash
python -m validation.validate_permutation
python -m validation.validate_addition --max_length 4 --pairs_per_length 200
```

Các flags hữu ích:

- `--max_length`: giảm độ dài bit để chạy nhanh.
- `--pairs_per_length`: với addition, tránh duyệt toàn bộ $2^{2\ell}$ cặp (rất nặng khi \(\ell\) lớn).

## Ghi chú Windows: cài neural-tangents / TensorFlow

`neural_tangents` thường kéo theo `tensorflow` (qua `jax2tf`). Trên Windows, quá trình `pip install tensorflow` có thể fail do **Windows Long Paths**.

Nếu gặp lỗi kiểu “No such file or directory … enable-long-paths”:

1) Bật Long Paths trong Windows (Group Policy hoặc registry `LongPathsEnabled=1`).
2) Hoặc đặt workspace vào đường dẫn ngắn hơn (vd `C:\ws\proj`) để giảm độ dài path.

