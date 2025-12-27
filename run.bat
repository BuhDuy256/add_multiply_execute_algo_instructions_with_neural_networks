conda activate exact_learning
cd training
python experiment.py --k-list 5 10 15 --trials 1 --epochs 10000 --target-acc 0.999