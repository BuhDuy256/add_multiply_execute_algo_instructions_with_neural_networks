@echo off
REM ============================================
REM  Odd K Values Experiment Script
REM  Runs permutation experiments for K = 3, 5, 7, 9, 11, 13
REM  Target accuracy: 90% (delta=0.1)
REM ============================================

echo Starting Odd K experiments at %date% %time%
echo Results will be added to: binary_algos_upstream\training\results\overnight_run

cd /d "c:\Users\ASUS\Downloads\add_multiply_execute_algo_instructions_with_neural_networks\binary_algos_upstream\training"

REM Create results directory (if not exists)
if not exist "results\overnight_run" mkdir "results\overnight_run"

REM Run experiments for each K value
for %%k in (3 5 7 9 11 13) do (
    echo.
    echo ============================================
    echo Starting K=%%k at %date% %time%
    echo ============================================
    "c:\Users\ASUS\Downloads\add_multiply_execute_algo_instructions_with_neural_networks\.venv\Scripts\python.exe" experiment.py --k %%k --trials 1 --epochs 2500 --hidden_dim 10000 --delta 0.1 --device cuda --outdir results/overnight_run
    echo K=%%k completed at %date% %time%
)

echo.
echo ============================================
echo All Odd K experiments completed at %date% %time%
echo ============================================

REM Generate plot - This will pick up BOTH old (even) and new (odd) results
echo Updating plots...
"c:\Users\ASUS\Downloads\add_multiply_execute_algo_instructions_with_neural_networks\.venv\Scripts\python.exe" ..\plot_results.py

echo Done! Charts updated with new K values.
pause
