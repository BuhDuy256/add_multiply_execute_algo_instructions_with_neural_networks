@echo off
REM ============================================
REM  K=18, 19 Experiment Script
REM  Target accuracy: 90%
REM ============================================

echo Starting K=18, 19 experiment at %date% %time%
echo Results will be added to: binary_algos_upstream\training\results\overnight_run

cd /d "c:\Users\ASUS\Downloads\add_multiply_execute_algo_instructions_with_neural_networks\binary_algos_upstream\training"

for %%k in (18 19) do (
    echo.
    echo ============================================
    echo Starting K=%%k at %date% %time%
    echo ============================================
    "c:\Users\ASUS\Downloads\add_multiply_execute_algo_instructions_with_neural_networks\.venv\Scripts\python.exe" experiment.py --k %%k --trials 1 --epochs 2500 --hidden_dim 10000 --delta 0.1 --device cuda --outdir results/overnight_run
    echo K=%%k completed at %date% %time%
)

echo.
echo ============================================
echo Updating plots...
"c:\Users\ASUS\Downloads\add_multiply_execute_algo_instructions_with_neural_networks\.venv\Scripts\python.exe" ..\plot_results.py

echo Done! Charts updated with K=18, 19.
pause
