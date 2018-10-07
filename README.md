# Neurify (NIPS'18)
Neurify is an efficient formal verification system for analyzing self-defined properties on given neural networks. It leverages symbolic linear relaxations based on symbolic interval analysis to provide tight output approximations. For cases unproved, it can further use linear solver to cut down false positives. In general, it is fast and can scale to large networks (e.g., over 10,000 ReLUs).   

You can find detailed description of Neurify in paper [Efficient Formal Safety Analysis of Neural Networks](https://arxiv.org/abs/1809.08098). Neurify is a followup paper upon a previous state-of-the-art verification system ReluVal. You can find the detailed description of symbolic interval analysis in paper [Formal Security Analysis of Neural Networks using Symbolic Intervals](https://arxiv.org/pdf/1804.10829.pdf).

This repository contains the implementation of Neurify and the evalutions on convolutional MNIST models, convolutional DAVE models, ACAS Xu models and Drebin models described in the paper.


## Prerequisite


### OpenBLAS Installation
OpenBLAS library is used for matrix multiplication speedup. So please make sure you have successfully installed [OpenBLAS](https://www.openblas.net/). You can follow following commands to install openblas or find the quick installation guide at [OpenBLAS's Installation Guide](https://github.com/xianyi/OpenBLAS/wiki/Installation-Guide).

```
wget http://github.com/xianyi/OpenBLAS/archive/v0.2.20.tar.gz
tar -xzf SOpenBLAS-0.2.20.tar.gz
cd OpenBLAS-0.2.20
make
make PREFIX=/path/to/your/installation install
```

### Downloading

```
git clone https://github.com/tcwangshiqi-columbia/Neurify
```

### Compiling:
Please make sure the path of OpenBLAS is the same as the one in MakeFile. Then you can compile Neurify with following command:

```
cd Neurify
cd convolutional
make
```

## File Structure

* network_test.c: main file to run with
* nnet.c: deal with network instance and do symbolic interval analysis
* split.c: manage iterative refinement and dynamic thread rebalancing
* matrix.c: matrix operations supported by OpenBLAS
* models(nnet)/: all the models
* scripts/: scripts to run the ACAS Xu evaluations reported in paper 

## Running 

The main function is in network_test.c. To run the function, you can call the binary ./network_test. It expects at least three arguments. Here is the argument list:

property: the saftety property want to verify

network: the network want to test with

need to print = 0: whether need to print the detailed information of each split. 0 is not and 1 is yes. Default value is 0.

test for one run = 0: whether need to estimate the output range without split refinement. 0 is no, 1 is yes. Default value is 0.

check mode = 0: choose the mode of formal anlysis. Normal split and check mode is 0. Check adv mode is 1. Check adv mode will prevent further splits as long as the bisection depth goes upper than 20 so as to locate concrete adversarial examples faster. Default value is 0.

The program will terminate in three ways: (1) a concrete adversarial is found, and (2) the property is verified as safe, and (3) Neurify hits predifined depth threshold indicating timeouts.

### Example

Here is an example for running Neurify:

```
./network_test 0 models/conv.nnet
```

### Properties

* The MNIST properties are defined as the classfier will not misclassify the given images bounded by L-1, L-2 and L-infinite. 
* The DAVE properties are defined as the classifier will predict correct steering angle (e.g., variance from original angle is less than 30 degree).
* The Drebin properties are defined as the classifier will still detect malware applications in terms of how many features are allowed to be given. 
* The ACAS Xu properties are reported and defined in ReluVal. One can find them in  Appendix A.


### Convolutional Model Experiments

The test on MNIST or DAVE models can be easily ran with commands. Here is an example:

```
cd convolutional
./network_test 0 models/conv_madry.nnet
./network_test 500 models/dave_small.nnet
```

### Drebin Model Experiments

The test on Drebin models can be easily ran with commands. Here is an example:

```
cd drebin
./network_test 101 models/drebin.nnet
```

### ACAS Xu Experiments

The test on ACAS Xu can be easily ran with pre-written scripts in folder "scripts". Here is an example:

```
cd ACAS
./scripts/run_property5.sh
```

### Custom Properties

One can customized their own properties, models and inputs. The properties can predefined in network.c and check function for each property can be added in split.c. For self-trained models, currently we support to convert the model trained with Keras or tensorflow to the format supported by Neurify by using transfer.py with configured model path. At last, self-defined inputs can be added into test set with unnormalized value and one can test them by updating the path in nnet.c.


## Citing Neurify
```
@inproceedings {Shiqi18,
	author = {Shiqi, Wang and Pei, Kexin and Justin, Whitehouse and Yang, Junfeng and Jana, Suman},
	title = {Efficient Formal Safety Analysis of Neural Networks},
	booktitle = {32nd Conference on Neural Information Processing Systems (NIPS)},
	year = {2018},
	address = {Montreal, Canada}
}
```


## Contributors

* [Shiqi Wang](https://sites.google.com/view/tcwangshiqi) - tcwangshiqi@cs.columbia.edu
* [Kexin Pei](https://sites.google.com/site/kexinpeisite/) - kpei@cs.columbia.edu
* [Justin Whitehouse](https://www.college.columbia.edu/node/11475) - jaw2228@columbia.edu
* [Junfeng Yang](http://www.cs.columbia.edu/~junfeng/) - junfeng@cs.columbia.edu
* [Suman Jana](http://www.cs.columbia.edu/~suman/) - suman@cs.columbia.edu


## License
Copyright (C) 2018-2019 by its authors and contributors and their institutional affiliations under the terms of modified BSD license.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.