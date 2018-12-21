cs231n module from https://github.com/cs231n/cs231n.github.io/blob/master/assignments/2018/spring1718_assignment1.zip

There are few modifications to fit in my local python environment.
* imread of scipy.misc is replaced by imageio due to the deprecation
* Define `load_CIFAR10_noreshape` which is loading CIFAR10 batch without reshape to reduce memory usages. The original `load_CIFAR10` causes MemoryError exception.
* Use float32 as CIFAR10 datatype.