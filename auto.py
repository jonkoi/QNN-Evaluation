import os
import subprocess

string = "python main_ABC.py --model ABC_cifar10 --save exp1_ABC_11 --dataset cifar10 --device 0"
print(string)
subprocess.call(string, shell=True)

string = "python mainW.py --model WAGE_cifar10 --save exp1_WAGE_11 --dataset cifar10 --device 0"
print(string)
subprocess.call(string, shell=True)

string = "python main.py --model BC_cifar10 --save exp1_BC_10 --dataset cifar10 --device 0"
print(string)
subprocess.call(string, shell=True)

string = "python main.py --model BNN_cifar10 --save exp1_BNN_10 --dataset cifar10 --device 0"
print(string)
subprocess.call(string, shell=True)

string = "python main.py --model BWN_cifar10 --save exp1_BWN_10 --dataset cifar10 --device 0"
print(string)
subprocess.call(string, shell=True)

string = "python main.py --model XNOR_cifar10 --save exp1_XNOR_10 --dataset cifar10 --device 0"
print(string)
subprocess.call(string, shell=True)

string = "python main.py --model Dorefa_cifar10 --save exp1_Dorefa_10 --dataset cifar10 --device 0"
print(string)
subprocess.call(string, shell=True)

string = "python main.py --model TWN_cifar10 --save exp1_TWN_10 --dataset cifar10 --device 0"
print(string)
subprocess.call(string, shell=True)

string = "python main.py --model TTQ_cifar10 --save exp1_TTQ_10 --dataset cifar10 --device 0"
print(string)
subprocess.call(string, shell=True)
