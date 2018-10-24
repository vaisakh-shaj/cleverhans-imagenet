# Cleverhans on Imagenet Architectures

Using [cleverhans](https://github.com/tensorflow/cleverhans) adversarial machine library for different Imagenet neural network architechtures implemented in tensorflow. Weights converted from caffemodels. Some weights were converted using `misc/convert.py` others using [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow). The weights can be downloaded from [here](https://www.dropbox.com/sh/qpuqj03gv00ba85/AAApqsIe4SqSOrsfpwrYjOema?dl=0). Tested with Tensorflow 1.0. Weights for inception-V3 taken from Keras implementation provided [here](https://github.com/fchollet/deep-learning-models/blob/master/inception_v3.py). Contributions are welcome!

## Features

* A single call program to create attacks on different architechtures (vgg-f, caffenet, vgg-16, vgg-19, googlenet, resnet-50, resnet-152, inception-V3) cleverhans.
* Can be extended to any attack that cleverhans support in future with few lines of changes in the code.

## Usage

* For creating attack on first 100 ilsvrc validation set images, `python ceverhans_attack.py --network 'resnet152' --attack 'fgsm' --sample_size 100`

* Currently the `--network` argument can take vggf, caffenet, vgg16, vgg19, googlenet, resnet50, resnet152, inceptionv3.

* Currently the `--attack` argument can take fgsm, ifgsm, pgd, deepfool, jsma, cw2.
