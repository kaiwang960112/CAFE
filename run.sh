##### reproduce dc results#########
python distill.py  --dataset CIFAR10  --model ConvNet  --ipc 10


########cross arch######
python distill.py  --dataset CIFAR10  --model ConvNet  --ipc 50  --eval_mode M