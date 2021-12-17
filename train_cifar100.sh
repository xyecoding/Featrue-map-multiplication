CUDA_VISIBLE_DEVICES=2,3 python  main.py --lr 0.1  --net_type wide-resnet --depth 16 --widen_factor 10 --dropout 0.3 --dataset Caltech101  2>&1 | tee log_100.txt
