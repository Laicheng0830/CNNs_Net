# CNNs_Net
Create the CNN mainstream network framework through Tensorflow, including LeNe, VGG, AlexNet,GoogleNet series(InceptionV1-V4); Test using MNIST and CIFAR data

CIFAR data:https://www.cs.toronto.edu/~kriz/cifar.html

Create CNNs network,Error encountered:

1.sess ,The late the better,Compile the model first, then initialize sess

2.The last layer, the full connection layer, does not use activation functions

3.learnrate,Too big: loss explodes, or nan,If the learning rate is set too high, it will cause problems of running and flying (loss suddenly remains big);too small: half a day loss is not reflected (however, it is also the case that LR needs to be reduced. Here, the intermediate results of visualization network are not weights, which have effects. The visualized results of the two are different.

4.bathsize,If the batchsize is too small, it will not converge

5.model pruning use tensorflow , model train modifier fully connected.
