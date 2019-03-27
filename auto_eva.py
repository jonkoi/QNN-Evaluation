import os
import subprocess

string = "python evaluate_standalone.py --model_name cifar10 --checkpoint_dir ./results/exp1_cifar10_final_3 --dataset cifar10 "
print(string)
subprocess.call(string, shell=True)

string = "python evaluate_standalone.py --model_name BNN_cifar10 --checkpoint_dir ./results/exp1_BNN_1 --dataset cifar10 "
print(string)
subprocess.call(string, shell=True)

string = "python evaluate_standalone.py --model_name BWN_cifar10 --checkpoint_dir ./results/exp1_BWN_1 --dataset cifar10 "
print(string)
subprocess.call(string, shell=True)

string = "python evaluate_standalone.py --model_name XNOR_cifar10 --checkpoint_dir ./results/exp1_XNOR_1 --dataset cifar10 "
print(string)
subprocess.call(string, shell=True)

string = "python evaluate_standalone.py --model_name Dorefa_cifar10 --checkpoint_dir ./results/exp1_Dorefa1132_2 --dataset cifar10 "
print(string)
subprocess.call(string, shell=True)

string = "python evaluate_standalone.py --model_name TWN_cifar10 --checkpoint_dir ./results/exp1_TWN_1 --dataset cifar10 "
print(string)
subprocess.call(string, shell=True)

string = "python evaluate_standalone.py --model_name TTQ_cifar10 --checkpoint_dir ./results/exp1_TTQ_1 --dataset cifar10 "
print(string)
subprocess.call(string, shell=True)

string = "python evaluate_standalone.py --model_name WAGE_cifar10 --checkpoint_dir ./results/exp1_WAGE_10 --dataset cifar10 "
print(string)
subprocess.call(string, shell=True)

string = "python evaluate_standalone.py --model_name ABC_cifar10 --checkpoint_dir ./results/exp1_ABC_11 --dataset cifar10 "
print(string)
subprocess.call(string, shell=True)
