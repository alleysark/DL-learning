# Biological motivation and connections
The basic computational unit of the brain is a **neuron**. In mathematical model, neuron receives several inputs from the previousely connected neurons *x_i* multiplied by synaptic strength *w_i*. It sums up all the inputs with bais *sigma{w_i * x_i} + b*. If this is above a certain threshold, the neuron can fire a spike along its axon. It is represented as an **activation function**. 

<img src="http://cs231n.github.io/assets/nn1/neuron.png" height=200 />
<img src="http://cs231n.github.io/assets/nn1/neuron_model.jpeg" height=200 />

# Single neuron as a linear classifier
A neuron has the capacity to "like" (activation near one) or "dislike" (activation near zero) certain linear regions of its input space. Hence, with an appropriate loss function on the neuron's output, we can turn a single neuron into a linear classifier.

# Activation functions
**sigmoid**: Sigmoid non-linearity squashes real numbers to range between [0, 1]. In practice, it is rarely ever used due to it saturates and kills gradients. If the local gradient is very small, no signal will flow through the neuron to its weights and recursively to its data.

**tanh**: tanh non-linearity squashes real numbers to range between [-1, 1]. Its activations also saturate, but unlike the sigmoid its output is zero-centered. It is simply a scaled sigmoid neuron. In practice the tanh non-linearity is always preferred to the sigmoid nonlinearity.

**ReLU**: Rectified Linear Unit *f(x)=max(0,x)*. It is the most popular activation function recently.
* (+) It accelerate the convergence of stochastic gradient descent compared to the sigmoid/tanh. It is argued that this is due to its linear, non-saturating form.
* (+) Computational simplicity is much greater than others.
* (-) ReLU units can be fragile during training if the learning rate is set too high. With a proper setting of the learningg rate this is less frequently an issue.

**Leaky ReLU**: These are one attempt to fix the "dying ReLU" problem. *f(x) = min(x, 0) * ax + max(x, 0) * x*, where *a* is a small constant. There are some successive reports with this, but the results are not always consistent.

**Maxout**: It generalizes the ReLU and its leaky version: *max(w_1 * x + b_1, w_2 * x + b_2)*. Maxout has all benefits of a ReLU unit and does not have its drawbacks (dying ReLU). However, it doubles the number of parameters for every single neuron, leading to a high total number of parameters.

> Use **ReLU** non-linearity with careful configurations of learning rates and monitoring the fraction of "dead" units in a network.

# Setting number of layers and their size
In practice it is often the case that 3-layer neural networks will outperform 2-layer nets, but going even deeper(4~ layer) rarely helps much more. This is in contrast to Convolutional Networks, since the images contain hierarchical structures.

If we increase the size and number of layers, the **capacity** ofthe network increases since the neurons can collaborate to express many different functions.
The rich expression of neural networks means that the network can learn to classify more complicated data. But it also makes the network overfit the training data.

It seams that smaller networks can be preferred for noncomplex data. However, smaller networks make training with local methods (such as gradient descent) harder. In practice, it is always better to use overfitting prevention methods (L2 regularization, dropout, input noise) to control overfitting instead of the number of neurons.

