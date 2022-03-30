import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

os.system("python3 main.py -s 256 --batch-size 16\
        --epochs 1000 --lr 1e-4 --gpu mars0\
        --flag '3rdLayer_FullLoss'\
        --memo 'Baseline\n'")
