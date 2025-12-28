import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def plot_experiment1(results_dir, output_path=None):
    """
    Experiment 1: Ensemble size to achieve target accuracy
    Biểu đồ: K value (trục x) vs Required N (trục y, log scale)
    """
    k_values = []
    required_n_values = []
    
    # Đọc dữ liệu từ summary.csv của mỗi K
    for k_dir in sorted(Path(results_dir).glob("k_*")):
        k = int(k_dir.name.split("_")[1])
        summary_file = k_dir / "summary.csv"
        
        if summary_file.exists():
            df = pd.read_csv(summary_file)
            # Trung bình required_n của tất cả trials
            avg_n = df['required_n'].mean()
            k_values.append(k)
            required_n_values.append(avg_n)
    
    if not k_values:
        print("Không tìm thấy dữ liệu! Chạy experiment trước.")
        return
    
    # Vẽ biểu đồ
    plt.figure(figsize=(8, 6))
    plt.semilogy(k_values, required_n_values, 'o-', linewidth=2, markersize=8, label='Empirical')
    
    # Có thể thêm theoretical bound nếu có
    # plt.semilogy(k_values, theoretical_values, '--', linewidth=2, label='Theoretical bound')
    
    plt.xlabel('Input size k', fontsize=12)
    plt.ylabel('Ensemble size', fontsize=12)
    plt.title('Experiment 1\nEnsemble size to achieve target accuracy', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu Experiment 1 vào: {output_path}")
    else:
        plt.show()
    
    plt.close()

def plot_experiment2(results_dir, output_path=None, max_n=300):
    """
    Experiment 2: Accuracy vs ensemble size
    Biểu đồ: Ensemble size (trục x) vs Accuracy (trục y)
    Mỗi K value là 1 đường màu khác nhau
    """
    plt.figure(figsize=(8, 6))
    
    colors = ['#2E8B57', '#4169E1', '#FF8C00', '#DC143C', '#9370DB', '#2F4F4F']
    k_dirs = sorted(Path(results_dir).glob("k_*"))
    
    for idx, k_dir in enumerate(k_dirs):
        k = int(k_dir.name.split("_")[1])
        progress_file = k_dir / "accuracy_progress.csv"
        
        if progress_file.exists():
            df = pd.read_csv(progress_file)
            
            # Lọc trial đầu tiên (hoặc average tất cả trials)
            trial_1 = df[df['trial_id'] == 1]
            
            if len(trial_1) > 0:
                n_values = trial_1['n'].values
                acc_values = trial_1['accuracy'].values
                
                # Giới hạn số điểm hiển thị
                mask = n_values <= max_n
                n_values = n_values[mask]
                acc_values = acc_values[mask]
                
                color = colors[idx % len(colors)]
                plt.plot(n_values, acc_values, linewidth=2, label=f'k = {k}', color=color)
    
    plt.xlabel('Ensemble size', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Experiment 2\nAccuracy vs ensemble size', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim([0, 1.05])
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu Experiment 2 vào: {output_path}")
    else:
        plt.show()
    
    plt.close()

def plot_both_experiments(results_dir, output_dir=None):
    """
    Vẽ cả 2 biểu đồ cạnh nhau (như trong paper)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ===== Experiment 1 =====
    k_values = []
    required_n_values = []
    
    for k_dir in sorted(Path(results_dir).glob("k_*")):
        k = int(k_dir.name.split("_")[1])
        summary_file = k_dir / "summary.csv"
        
        if summary_file.exists():
            df = pd.read_csv(summary_file)
            avg_n = df['required_n'].mean()
            k_values.append(k)
            required_n_values.append(avg_n)
    
    if k_values:
        ax1.semilogy(k_values, required_n_values, 'o-', linewidth=2, 
                     markersize=8, label='Empirical', color='#4169E1')
        ax1.set_xlabel('Input size k', fontsize=12)
        ax1.set_ylabel('Ensemble size', fontsize=12)
        ax1.set_title('Experiment 1\nEnsemble size to achieve target accuracy', fontsize=13)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # ===== Experiment 2 =====
    colors = ['#2E8B57', '#4169E1', '#FF8C00', '#DC143C', '#9370DB', '#2F4F4F']
    k_dirs = sorted(Path(results_dir).glob("k_*"))
    
    for idx, k_dir in enumerate(k_dirs):
        k = int(k_dir.name.split("_")[1])
        progress_file = k_dir / "accuracy_progress.csv"
        
        if progress_file.exists():
            df = pd.read_csv(progress_file)
            trial_1 = df[df['trial_id'] == 1]
            
            if len(trial_1) > 0:
                n_values = trial_1['n'].values
                acc_values = trial_1['accuracy'].values
                
                # Giới hạn 300 points
                mask = n_values <= 300
                n_values = n_values[mask]
                acc_values = acc_values[mask]
                
                color = colors[idx % len(colors)]
                ax2.plot(n_values, acc_values, linewidth=2, 
                        label=f'k = {k}', color=color)
    
    ax2.set_xlabel('Ensemble size', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Experiment 2\nAccuracy vs ensemble size', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / "both_experiments.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu cả 2 biểu đồ vào: {output_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot experiment results')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Thư mục chứa kết quả (ví dụ: results/combined_run_20231228_120000)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Thư mục lưu ảnh output (mặc định: results-dir)')
    parser.add_argument('--plot-type', type=str, default='both',
                       choices=['exp1', 'exp2', 'both'],
                       help='Loại biểu đồ: exp1, exp2, hoặc both (mặc định: both)')
    parser.add_argument('--max-n', type=int, default=300,
                       help='Số models tối đa hiển thị trong Experiment 2')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.plot_type == 'exp1':
        plot_experiment1(results_dir, output_dir / "experiment1.png")
    elif args.plot_type == 'exp2':
        plot_experiment2(results_dir, output_dir / "experiment2.png", args.max_n)
    else:  # both
        plot_both_experiments(results_dir, output_dir)
        # Cũng lưu riêng lẻ
        plot_experiment1(results_dir, output_dir / "experiment1.png")
        plot_experiment2(results_dir, output_dir / "experiment2.png", args.max_n)

if __name__ == "__main__":
    main()
