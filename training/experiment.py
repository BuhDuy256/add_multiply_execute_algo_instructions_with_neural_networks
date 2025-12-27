import torch
import torch.optim as optim
import argparse
import os
import csv
from datetime import datetime
from pathlib import Path
import dataset
from model import NTKMLP
from tqdm import tqdm

# Tự động chọn GPU nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_single_model(train_loader, k, hidden_dim, epochs=2500, lr=0.1):
    """
    Huấn luyện một mô hình NTKMLP trên basis vectors.
    Kiến trúc mạng 2 lớp không bias.
    """
    model = NTKMLP(k, hidden_dim, k).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    model.train()
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return model.eval()

def get_accuracy(all_preds, Y_truth):
    """
    Tính accuracy bằng Ensemble Mean G(x) > 0.
    """
    ensemble_mean = torch.stack(all_preds).mean(dim=0)
    predictions = (ensemble_mean > 0).float()
    return (predictions == Y_truth.to(device)).all(dim=1).float().mean().item()

def run_combined_experiment(k_list, target_acc, trials, epochs, output_dir, hidden_dim_override=None):
    """
    Thực hiện song song:
    1. Thí nghiệm 1: Tìm N tối thiểu để đạt target_acc.
    2. Thí nghiệm 2: Lưu dữ liệu tăng trưởng Accuracy theo từng n để vẽ đường cong.
    
    Args:
        hidden_dim_override: Nếu được cung cấp, dùng giá trị này thay vì k*1000 cho TẤT CẢ K values
    """
    print(f"\n{'='*60}")
    print(f"Thí nghiệm Tổng hợp: Required N & Accuracy Curve")
    print(f"Target Accuracy: {target_acc}")
    print(f"Trials per K: {trials}")
    print(f"{'='*60}\n")
    
    for k in k_list:
        print(f"\nProcessing K = {k}")
        # Cho phép override hidden_dim, nếu không thì dùng k * 1000
        hidden_dim = hidden_dim_override if hidden_dim_override is not None else k * 1000
        print(f"Hidden dim: {hidden_dim}")
        
        # Tạo folder cho K này
        k_dir = output_dir / f"k_{k}"
        k_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. File summary.csv: Chỉ lưu điểm dừng (Required N) của mỗi trial
        summary_path = k_dir / "summary.csv"
        with open(summary_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['trial_id', 'required_n', 'final_accuracy'])
        
        # 2. File accuracy_progress.csv: Lưu TOÀN BỘ quá trình (dùng cho Thí nghiệm 2)
        progress_path = k_dir / "accuracy_progress.csv"
        with open(progress_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['trial_id', 'n', 'accuracy'])
        
        n_per_trial = []
        last_permutation = None
        
        for t in range(trials):
            P = dataset.generate_permutation(k) #
            last_permutation = P
            
            train_loader = torch.utils.data.DataLoader(
                dataset.PermutationDataset(P), 
                batch_size=k
            )
            
            # Tạo test set với nnz=2
            X_test_norm, X_test_raw = dataset.generate_test_set(k, num_samples=200, nnz=2)
            Y_test = (P @ X_test_raw.T).T.to(device)
            X_test_norm = X_test_norm.to(device)
            
            all_preds = []
            acc = 0
            n = 0
            
            # Progress bar cho từng Trial
            pbar = tqdm(total=None, desc=f"  K={k} Trial {t+1}/{trials}", unit="model")
            
            # Mở file progress để append dữ liệu liên tục (đề phòng crash)
            with open(progress_path, 'a', newline='', encoding='utf-8') as f_prog:
                prog_writer = csv.writer(f_prog)
                
                while acc < target_acc and n < 5000:
                    n += 1
                    model = train_single_model(train_loader, k, hidden_dim, epochs)
                    
                    with torch.no_grad():
                        pred = model(X_test_norm).detach()
                        all_preds.append(pred)
                    
                    acc = get_accuracy(all_preds, Y_test)
                    
                    # GHI DỮ LIỆU THÍ NGHIỆM 2: Lưu accuracy tại mỗi bước n
                    prog_writer.writerow([t + 1, n, f'{acc:.6f}'])
                    
                    pbar.update(1)
                    pbar.set_postfix({'acc': f'{acc:.4f}'})
                    
                    if acc >= target_acc:
                        break
            
            pbar.close()
            n_per_trial.append(n)
            
            # GHI DỮ LIỆU THÍ NGHIỆM 1: Lưu Required N của trial
            with open(summary_path, 'a', newline='', encoding='utf-8') as f_sum:
                sum_writer = csv.writer(f_sum)
                sum_writer.writerow([t + 1, n, f'{acc:.6f}'])
            
            print(f"    -> Trial {t+1} hoàn tất tại N = {n}")

        # Lưu thống kê cuối cùng cho K
        avg_n = sum(n_per_trial) / trials
        with open(k_dir / "final_results.txt", 'w') as f:
            f.write(f"Avg Required N: {avg_n:.2f}\n")
            f.write(f"Trials: {trials}\n")
            f.write(f"Epochs: {epochs}\n")
        
        if last_permutation is not None:
            torch.save(last_permutation, k_dir / "permutation.pt")

def main():
    parser = argparse.ArgumentParser(description='NTK Combined Experiment')
    parser.add_argument("--k-list", type=int, nargs="+", default=[5, 10, 15, 20],
                        help="List of K values to test")
    parser.add_argument("--trials", type=int, default=5,
                        help="Number of trials per K value")
    parser.add_argument("--epochs", type=int, default=1250,
                        help="Number of epochs per model training")
    parser.add_argument("--target-acc", type=float, default=0.9,
                        help="Target accuracy to reach (default: 0.9)")
    parser.add_argument("--hidden-dim", type=int, default=None,
                        help="Fixed hidden dimension for all K values. If not specified, uses k*1000")
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results") / f"combined_run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Dữ liệu được lưu tại: {output_dir}")
    if args.hidden_dim:
        print(f"Using fixed hidden_dim = {args.hidden_dim} for all K values")
    else:
        print(f"Using auto-scaling hidden_dim = k * 1000")
    
    run_combined_experiment(args.k_list, args.target_acc, args.trials, args.epochs, 
                           output_dir, args.hidden_dim)

if __name__ == "__main__":
    main()