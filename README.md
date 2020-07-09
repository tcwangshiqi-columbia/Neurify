# Neurify (NIPS'18)
Neurify is an efficient formal verification system for analyzing self-defined properties on given neural networks. It leverages symbolic linear relaxations based on symbolic interval analysis to provide tight output approximations. For cases unproved, it can further use linear solver to cut down false positives. In general, it is able to scale to large networks (e.g., over 10,000 ReLUs).   

You can find detailed descriptions of Neurify in paper [Efficient Formal Safety Analysis of Neural Networks](https://arxiv.org/abs/1809.08098). Neurify is a followup paper upon a previous state-of-the-art verification system ReluVal. Detailed descriptions of symbolic interval analysis in paper [Formal Security Analysis of Neural Networks using Symbolic Intervals](https://arxiv.org/pdf/1804.10829.pdf).

This repository contains the implementations of Neurify and the evalutions on convolutional MNIST models, convolutional DAVE models and Drebin models described in the paper. The updates on ACAS Xu model have been merged into original [ReluVal's repo](https://github.com/tcwangshiqi-columbia/ReluVal). Neurify's performance on ACAS Xu is on average 20 times better than original ReluVal's and 5000 times better than solver-based system like Reluplex.

We highly recommend users to compile and run the code under Linux 16.04 in case any potential dependency or compilation errors.

[News] PyTorch version of symbolic interval analysis (supporting gpus) is now available at our [Symbolic Interval Library](https://github.com/tcwangshiqi-columbia/symbolic_interval).

[News] We have updated the interface and code structure in "general" file such that people can customize Neurify in an easier way. It replaces the original mnist folder.

[News] Neurify code has been cleaned up thanks to [Christopher Brix](https://www.christopher-brix.de/)!

## Prerequisite


### OpenBLAS Installation
OpenBLAS library is used for matrix multiplication speedup. So please make sure you have successfully installed [OpenBLAS](https://www.openblas.net/). You can follow following commands to install openblas or find the quick installation guide at [OpenBLAS's Installation Guide](https://github.com/xianyi/OpenBLAS/wiki/Installation-Guide).

```
wget http://github.com/xianyi/OpenBLAS/archive/v0.2.20.tar.gz
tar -xzf v0.2.20.tar.gz 
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
cd Neurify/general
make
```

## File Structure

* network_test.c: main file to run with
* nnet.c: deal with network instance and do symbolic interval analysis
* split.c: manage iterative refinement and dynamic thread rebalancing
* matrix.c: matrix operations supported by OpenBLAS
* models/: all the models

## Running 

The main function is in network_test.c. To run the function, you can call the binary ./network_test. It expects at least three arguments. Here is the argument list:

property: the saftety property want to verify

network: the network want to test with

need to print = 0: whether need to print the detailed information of each split. 0 is not and 1 is yes. Default value is 0.

test for one run = 0: whether need to estimate the output range without split refinement. 0 is no, 1 is yes. Default value is 0.

check mode = 0: choose the mode of formal anlysis. Normal split and check mode is 0. Check adv mode is 1. Check adv mode will prevent further splits as long as the bisection depth goes upper than 10 so as to locate concrete adversarial examples faster. Default value is 0.

MAX_DEPTH: A flexible parameter to adjust the timeout threshold.

norm_input: Whether needs to normalize the input.

The program will terminate in three ways: (1) a concrete adversarial is found, and (2) the property is verified as safe, and (3) Neurify hits predifined depth threshold indicating timeouts.

### Usage

Here is an example for running Neurify by checking whether convolutional MNIST model (models/conv.nnet) will always predict correct labels within input ranges bounded by infinite norm (property 0) for given testing images (one can select images in network_test.c):

```
./network_test 0 models/conv.nnet
```

Please check network.c to see the detailed descriptions of optional input parameters. 

Note that there are two different modes of Neurify. One is regular mode. It will return (1) the input is verified safe, (2) verified unsafe by locating concrete adversarial examples, (3) time out. One can preset the MAX_DEPTH to adjust the timeout threshold. The second mode is CHECK_ADV_MODE. It will do a breath-first search with shallow MAX_DEPTH aiming to locate the adversarial examples quickly. Compared to the regular mode, it can find the concrete adversarial examples in a much faster way, but no verification for safety can be provided under this mode. Therefore, it's recommended to first run the CHECK_ADV_MODE quickly. If no adversarial examples can be located for the specific input, then we can use the regular mode with large MAX_DEPTH to verify its safety.


### Properties

One can customized their own properties, models and inputs. The whole project is updated and merged into file "general" such that people can easily customize Neurify for their own purpose. To customize, one needs to update (1) the input loading function in nnet.c, (2) property checking functions in split.c, and (3) input ranges of the safety property in network.c.

Here are the property list that Neurify can supported. 0: MNIST L-infinite norm; 54: MNIST L-1 norm; 55: MNIST brightness; 101: Drebin; 500: DAVE L-infinite norm; 504: DAVE L-1 norm; 505: DAVE brightness; 510: DAVE contrast.

* The MNIST properties are defined as the classfier will not misclassify the given images bounded by L-1, L-2 and L-infinite. 
* The DAVE properties are defined as the classifier will predict correct steering angle (e.g., variance from original angle is less than 30 degree).
* The Drebin properties are defined as the classifier will still detect malware applications in terms of how many features are allowed to be given. 
* The ACAS Xu properties are reported and defined in ReluVal. One can find them in Appendix A.


### Convolutional Model Experiments

The test on MNIST can be easily ran with commands. Here is an example:

```
cd general
./network_test 0 models/conv_madry.nnet
```

### Drebin Model Experiments

The test on Drebin models can be easily ran with commands. Here is an example:

```
cd drebin
./network_test 101 models/drebin.nnet
```

### ACAS Xu Experiments

Please clone the [ReluVal's repo](https://github.com/tcwangshiqi-columbia/ReluVal). The test on ACAS Xu can be easily ran with pre-written scripts in folder "scripts". Here is an example:

```
cd ACAS
./scripts/run_property5.sh
```

### Supporting Other ML Frameworks

Please check the nnet model format in "format descriptions" file for more details. In transfer.py, we provide a sample file to transfer tensorflow model to nnet format.

In file "docker", you can find a dockerfile and requirements for running Neurify, and sample scripts for transferring keras models to nnet format (credit to [Jonas Klamroth](https://www.fzi.de/en/about-us/organisation/detail/address/klamroth/)).

## Citing Neurify
```
@inproceedings {shiqi2018neurify,
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
* [Christopher Brix](https://www.christopher-brix.de/) - Christopher.Brix@rwth-aachen.de

## License
Copyright (C) 2018-2019 by its authors and contributors and their institutional affiliations under the terms of modified BSD license.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
