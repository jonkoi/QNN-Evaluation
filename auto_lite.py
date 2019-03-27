import os
import subprocess


for i in range(1,13):
    a= os.listdir('./'  + str(i) + '/')
    print('START: ' + str(i))
    for i in range(10):
        b = './'  + str(i) + '/' + a[i]
        string = "./label_image_2 -m exp2.tflite -i " + b + " -l labels_exp2.txt"
        print(string)
        subprocess.call(string, shell=True)
