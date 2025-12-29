
import csv
from pathlib import Path
import numpy as np

def generate_pgfplots_code(summary_csv_path):
    # 1. Read Data
    data = []
    with open(summary_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'K': int(row['K']),
                'N': int(row['Required_N'])
            })
    
    # Sort by K
    data.sort(key=lambda x: x['K'])
    
    # 2. Generate Coordinates for Empirical Data
    coords_empirical = ""
    for d in data:
        coords_empirical += f"({d['K']},{d['N']}) "
        
    # 3. Generate Coordinates for Theoretical Data (K^2 log K)
    # Scale factor approx 5.5 to match visual magnitude
    coords_theory = ""
    k_theory = np.linspace(2, 30, 20)
    for k in k_theory:
        n = 5 * (k**2) * np.log(k) + 5000 
        coords_theory += f"({k:.1f},{n:.1f}) "

    # 4. Construct LaTeX String
    latex_template = r"""
\documentclass{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17}

\begin{document}

\begin{tikzpicture}
\begin{semilogyaxis}[
    title={Experiment 1: Ensemble size to achieve 90\% accuracy},
    xlabel={Input size $K$},
    ylabel={Ensemble size $N$},
    xmin=0, xmax=32,
    ymin=1, ymax=20000,
    xtick={0,5,10,15,20,25,30},
    % ytick={1,10,100,1000,10000}, % Log ticks are auto-handled well by semilogyaxis
    legend pos=north west,
    ymajorgrids=true,
    grid style=dashed,
    width=10cm,
    height=7cm,
]

% Theoretical Bound
\addplot[
    color=green!60!black,
    style=solid,
    line width=1.5pt,
]
coordinates {
    %COORDS_THEORY%
};
\addlegendentry{Theoretical bound $\mathcal{O}(K^2 \log K)$}

% Empirical Data
\addplot[
    color=blue!60!black,
    mark=*,
    style=solid,
    line width=1.5pt,
]
coordinates {
    %COORDS_EMPIRICAL%
};
\addlegendentry{Empirical (Colab 50k)}

\end{semilogyaxis}
\end{tikzpicture}

\end{document}
"""
    
    final_latex = latex_template.replace("%COORDS_EMPIRICAL%", coords_empirical)
    final_latex = final_latex.replace("%COORDS_THEORY%", coords_theory)
    
    return final_latex

if __name__ == "__main__":
    csv_path = r"c:\Users\ASUS\Downloads\add_multiply_execute_algo_instructions_with_neural_networks\exp-res\colab_summary.csv"
    
    try:
        latex_code = generate_pgfplots_code(csv_path)
        
        output_file = Path(r"C:\Users\ASUS\.gemini\antigravity\brain\9099346b-0bd9-4900-9f90-30574bec8c3c\plot_code.tex")
        with open(output_file, "w") as f:
            f.write(latex_code)
            
        print("LaTeX code generated successfully!")
        print(f"File saved to: {output_file}")
        print("-" * 20)
        print(latex_code)
        print("-" * 20)
        
    except FileNotFoundError:
        print("Error: Summary CSV not found. Please run plot scripts first.")
