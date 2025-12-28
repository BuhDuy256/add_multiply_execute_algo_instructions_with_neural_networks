conda activate exact_learning
cd training
python experiment.py --k-list 5 --trials 1 --epochs 6000 --target-acc 0.99 --hidden-dim 10000 --lr 0.001
python experiment.py --k-list 10 --trials 1 --epochs 6000 --target-acc 0.98 --hidden-dim 10000 --lr 0.001
python experiment.py --k-list 15 --trials 1 --epochs 6000 --target-acc 0.97 --hidden-dim 10000 --lr 0.001
python experiment.py --k-list 20 --trials 1 --epochs 6000 --target-acc 0.96 --hidden-dim 10000 --lr 0.001