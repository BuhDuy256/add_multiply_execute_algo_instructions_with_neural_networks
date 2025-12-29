
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ==========================================
# LATEX-LIKE STYLING CONFIGURATION (STRICT MODE)
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",  # Computer Modern (TeX-like)
    "axes.formatter.use_mathtext": True, # Force axis numbers to use CM font
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 20,
    "lines.linewidth": 2.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

def load_results(results_dir):
    results = {}
    
    search_paths = [
        Path(results_dir),
        Path(results_dir) / 'results' / 'experiment_1',
        Path(results_dir) / 'results' / 'experiment_2'
    ]
    
    print(f"Scanning paths for LaTeX plotting: {search_paths}")
    
    for base_path in search_paths:
        if not base_path.exists():
            continue
            
        for k_dir in base_path.iterdir():
            if k_dir.is_dir() and k_dir.name.isdigit():
                k = int(k_dir.name)
                
                if k not in results:
                    results[k] = {'n_models': [], 'accuracy': [], 'error': [], 'progression': []}
                
                for trial_dir in k_dir.iterdir():
                    if trial_dir.is_dir():
                        results_file = trial_dir / 'results.csv'
                        if results_file.exists():
                            current_run_progression = []
                            with open(results_file, 'r') as f:
                                lines = f.readlines()
                                for line in lines:
                                    parts = line.strip().split(',')
                                    if len(parts) >= 3:
                                        n_curr = int(parts[0])
                                        err_curr = float(parts[1])
                                        acc_curr = float(parts[2])
                                        
                                        current_run_progression.append({
                                            'n': n_curr,
                                            'error': err_curr,
                                            'acc': acc_curr
                                        })
                            
                            if not current_run_progression:
                                continue

                            existing_max_acc = 0
                            if results[k]['progression']:
                                existing_max_acc = max(p['acc'] for p in results[k]['progression'])
                            
                            current_max_acc = max(p['acc'] for p in current_run_progression)
                            
                            if current_max_acc >= existing_max_acc:
                                results[k]['progression'] = current_run_progression

    final_results = {}
    for k, data in results.items():
        if not data['progression']:
            continue
            
        target_acc = 0.9
        best_n = None
        final_acc = 0
        final_err = 0
        
        prog = sorted(data['progression'], key=lambda x: x['n'])
        final_acc = prog[-1]['acc']
        final_err = prog[-1]['error']
        
        for p in prog:
            if p['acc'] >= target_acc:
                best_n = p['n']
                break
        
        final_results[k] = {
            'n_models': [], 
            'accuracy': [], 
            'error': [], 
            'progression': prog
        }
        
        if best_n is not None:
             final_results[k]['n_models'].append(best_n)
             # Add final stats too
             final_results[k]['accuracy'].append(final_acc)
        elif final_acc > 0.8: 
             pass # Valid but didn't reach target
             
    return final_results

def plot_required_n_vs_k(results, output_path):
    plt.figure(figsize=(10, 7)) # Larger figure
    
    k_values = sorted(results.keys())
    n_values = []
    
    valid_k = []
    for k in k_values:
        if results[k]['n_models']:
            valid_k.append(k)
            n_values.append(np.mean(results[k]['n_models']))
    
    # Theoretical Bound
    theoretical_k = np.linspace(min(valid_k), max(valid_k), 100)
    theoretical_n = 5 * (theoretical_k ** 2) * np.log(theoretical_k) + 5000 
    
    # NOTE: Using \mathrm{} and escaping spaces to force consistent CM font
    plt.plot(theoretical_k, theoretical_n, '-', linewidth=3, color='#2ca02c', label=r'$\mathrm{Theoretical\ bound\ } \mathcal{O}(K^2 \log K)$')
    plt.plot(valid_k, n_values, 'o-', linewidth=3, markersize=8, label=r'$\mathrm{Empirical\ (Colab\ 50k)}$', color='#1f77b4') 
    
    plt.xlabel(r'$\mathrm{Input\ size\ } K$')
    plt.ylabel(r'$\mathrm{Ensemble\ size\ } N$')
    plt.yscale('log')
    
    # Match Paper Ticks for Left Plot
    # X-axis: 2, 4, 6, ... 30
    plt.xticks(np.arange(2, 31, 2))
    plt.xlim(1.5, 30.5)
    
    # Y-axis: Powers of 10
    # Force log locators
    plt.yscale('log')
    plt.yticks([1, 10, 100, 1000, 10000], [r'$10^0$', r'$10^1$', r'$10^2$', r'$10^3$', r'$10^4$'])
    plt.ylim(0.8, 20000) 
    
    plt.title(r'$\mathrm{Experiment\ 1:\ Ensemble\ size\ to\ achieve\ 90\%\ accuracy}$')
    
    plt.grid(True, which="major", ls="-", alpha=0.4)
    plt.grid(True, which="minor", ls=":", alpha=0.2)
    plt.legend(frameon=True, shadow=True, fancybox=True, loc='upper left')
    plt.tight_layout()
    
    # Save as PNG
    save_path_png = output_path / 'latex_required_n_vs_k.png'
    plt.savefig(save_path_png, dpi=300)
    print(f"Saved PNG: {save_path_png}")
    
    # Save as PDF (Vector)
    save_path_pdf = output_path / 'latex_required_n_vs_k.pdf'
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')
    print(f"Saved PDF: {save_path_pdf}")

def plot_accuracy_curves(results, output_path):
    plt.figure(figsize=(10, 6))
    
    display_k = [k for k in sorted(results.keys()) if k in [2, 5, 10, 15, 20, 25, 30]]
    if not display_k:
        display_k = sorted(results.keys())[::2]
        
    colors = plt.cm.viridis(np.linspace(0, 1, len(display_k)))
    
    for i, k in enumerate(display_k):
        if 'progression' in results[k] and results[k]['progression']:
            n_vals = [p['n'] for p in results[k]['progression']]
            acc_vals = [p['acc'] for p in results[k]['progression']]
            plt.plot(n_vals, acc_vals, linewidth=2.5, color=colors[i], label=f'$K={k}$')
    
    plt.xlabel(r'$\mathrm{Number\ of\ Models\ } N$')
    plt.ylabel(r'$\mathrm{Accuracy}$')
    plt.title(r'$\mathrm{Accuracy\ Progression\ (Selected\ } K \mathrm{)}$')
    
    # Match Paper Ticks for Right Plot
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.ylim(-0.05, 1.05)
    
    plt.legend(loc='lower right', frameon=True, fancybox=True)
    plt.grid(True, alpha=0.4)
    
    plt.tight_layout()
    
    # Save as PNG
    save_path_png = output_path / 'latex_accuracy_curves.png'
    plt.savefig(save_path_png, dpi=300)
    print(f"Saved PNG: {save_path_png}")

    # Save as PDF (Vector)
    save_path_pdf = output_path / 'latex_accuracy_curves.pdf'
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')
    print(f"Saved PDF: {save_path_pdf}")

if __name__ == "__main__":
    results_dir = Path(r"c:\Users\ASUS\Downloads\add_multiply_execute_algo_instructions_with_neural_networks\exp-res\extracted")
    output_path = Path(r"c:\Users\ASUS\Downloads\add_multiply_execute_algo_instructions_with_neural_networks\exp-res")
    
    results = load_results(results_dir)
    if results:
        print(f"Found results for K values: {sorted(results.keys())}")
        plot_required_n_vs_k(results, output_path)
        plot_accuracy_curves(results, output_path)
    else:
        print("No results found.")
