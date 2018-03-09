## Update
This implementation is based on https://github.com/itayhubara/BinaryNet.tf  
Changes or update frome orginal version
1. tensorflow dependency from 1.2.1 to 1.4.0 with related code update
2. adding logging function to implement local log recording
3. results is saved by time format directory rather than covering mode like orignial version
4. fix some bugs of orignial version to work well in tensorflow 1.4.0
5. fix incorrect cifar10 data preprocessing code
## BinaryNet.tf
Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1.  implementation in tensorflow (https://papers.nips.cc/paper/6573-binarized-neural-networks)

This is incomplete training example for BinaryNets using Binary-Backpropagation algorithm as explained in 
"Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, 
on following datasets: Cifar10/100.

Note that in this folder I didn’t implemented (yet...) shift-base BN , shift-base AdaMax (instead I just use the vanilla BN and Adam).
Likewise, I use deterministic binarization and I don't apply the initialization coefficients from GLorot&Bengio 2010.
Finally "sparse_softmax_cross_entropy_with_logits" loss is used instead if the SquareHingeLoss. 

The implementation is based on https://github.com/eladhoffer/convNet.tf but the main idea can be easily transferred to any tensorflow wrapper. I'll probably change it to keras soon. 
(e.g., slim,keras)

## Data
This implementation supports cifar10/cifar100  

## Dependencies
tensorflow version 1.2.1

## Training

* Train cifar10 model using gpu:
 python main.py --model BNN_vgg_cifar10 --save BNN_cifar10 --dataset cifar10 --gpu True
* Train cifar10 model using cpu:
 python main.py --model BNN_vgg_cifar10 --save BNN_cifar10 --dataset cifar10


## Results
Cifar10 should reach at least 88% top-1 accuracy






