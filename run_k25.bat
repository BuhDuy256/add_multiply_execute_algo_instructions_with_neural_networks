@echo off
REM ============================================
REM  K=25 Experiment Script
REM  Target accuracy: 90% (delta=0.1)
REM ============================================

echo Starting K=25 experiment at %date% %time%
echo (This may take a while, estimated 1000+ models)
echo Results will be added to: binary_algos_upstream\training\results\overnight_run

cd /d "c:\Users\ASUS\Downloads\add_multiply_execute_algo_instructions_with_neural_networks\binary_algos_upstream\training"

echo.
echo ============================================
echo Starting K=25 at %date% %time%
echo ============================================
"c:\Users\ASUS\Downloads\add_multiply_execute_algo_instructions_with_neural_networks\.venv\Scripts\python.exe" experiment.py --k 25 --trials 1 --epochs 2500 --hidden_dim 10000 --delta 0.1 --device cuda --outdir results/overnight_run
echo K=25 completed at %date% %time%

echo.
echo ============================================
echo Updating plots...
"c:\Users\ASUS\Downloads\add_multiply_execute_algo_instructions_with_neural_networks\.venv\Scripts\python.exe" ..\plot_results.py

echo Done! Charts updated.
pause
