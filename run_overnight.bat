@echo off
REM ============================================
REM  Overnight Experiment Script
REM  Runs permutation experiments for K = 2,4,6,8,10,15,20,30
REM  Target accuracy: 90% (delta=0.1)
REM ============================================

echo Starting overnight experiments at %date% %time%
echo Results will be saved to: binary_algos_upstream\training\results\overnight_run

cd /d "c:\Users\ASUS\Downloads\add_multiply_execute_algo_instructions_with_neural_networks\binary_algos_upstream\training"

REM Create results directory
if not exist "results\overnight_run" mkdir "results\overnight_run"

REM Run experiments for each K value
for %%k in (2 4 6 8 10 15 20 30) do (
    echo.
    echo ============================================
    echo Starting K=%%k at %date% %time%
    echo ============================================
    "c:\Users\ASUS\Downloads\add_multiply_execute_algo_instructions_with_neural_networks\.venv\Scripts\python.exe" experiment.py --k %%k --trials 1 --epochs 2500 --hidden_dim 10000 --delta 0.1 --device cuda --outdir results/overnight_run
    echo K=%%k completed at %date% %time%
)

echo.
echo ============================================
echo All experiments completed at %date% %time%
echo ============================================

REM Generate plot
echo Generating plot...
"c:\Users\ASUS\Downloads\add_multiply_execute_algo_instructions_with_neural_networks\.venv\Scripts\python.exe" ..\plot_results.py

echo Done! Check results\overnight_run\ for data and plot.
pause
