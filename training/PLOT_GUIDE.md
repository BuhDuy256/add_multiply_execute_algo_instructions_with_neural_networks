# Hướng dẫn vẽ biểu đồ kết quả

## Tổng quan

Script `plot_results.py` vẽ 2 biểu đồ từ kết quả thí nghiệm:

- **Experiment 1**: Số models cần thiết (N) để đạt target accuracy cho mỗi K
- **Experiment 2**: Accuracy tăng dần theo số models cho từng K value

## Cách sử dụng

### 1. Sau khi chạy experiment xong:

```bash
# Experiment sẽ tạo folder: results/combined_run_YYYYMMDD_HHMMSS/
python experiment.py --k-list 5 10 --trials 1 --epochs 5000 --target-acc 0.98 --hidden-dim 8000 --lr 0.001
```

### 2. Vẽ biểu đồ từ kết quả:

#### Vẽ cả 2 biểu đồ (mặc định):

```bash
python plot_results.py --results-dir results/combined_run_20231228_120000
```

#### Vẽ chỉ Experiment 1:

```bash
python plot_results.py --results-dir results/combined_run_20231228_120000 --plot-type exp1
```

#### Vẽ chỉ Experiment 2:

```bash
python plot_results.py --results-dir results/combined_run_20231228_120000 --plot-type exp2
```

#### Giới hạn số models hiển thị (Experiment 2):

```bash
python plot_results.py --results-dir results/combined_run_20231228_120000 --max-n 100
```

#### Chỉ định thư mục output:

```bash
python plot_results.py --results-dir results/combined_run_20231228_120000 --output-dir plots
```

## Output

Script sẽ tạo các file:

- `both_experiments.png` - Cả 2 biểu đồ cạnh nhau (giống paper)
- `experiment1.png` - Chỉ Experiment 1
- `experiment2.png` - Chỉ Experiment 2

## Ví dụ đầy đủ

```bash
# 1. Chạy experiment
python experiment.py --k-list 5 10 15 --trials 1 --epochs 4000 --target-acc 0.97 --hidden-dim 8000 --lr 0.001

# 2. Đợi cho xong, sau đó vẽ biểu đồ
python plot_results.py --results-dir results/combined_run_20231228_120000

# 3. Xem kết quả trong folder results/combined_run_20231228_120000/
```

## Lưu ý

- Đảm bảo đã cài `matplotlib` và `pandas`:

  ```bash
  pip install matplotlib pandas numpy
  ```

- Nếu không có dữ liệu, script sẽ báo "Không tìm thấy dữ liệu!"

- Experiment 2 mặc định hiển thị 300 models đầu tiên (có thể thay đổi với `--max-n`)
