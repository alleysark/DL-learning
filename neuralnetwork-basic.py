# code and its explanations from this post: https://hackernoon.com/dl02-writing-a-neural-network-from-scratch-code-b32f4877c257

import numpy as np
import dill #used to store all variables in a python file

class neural_network:
    # num_layers: number of layers in the network
    # num_nodes: list of size num_layers, specifying the number of nodes in each layer
    # activation_functions: activation function for each layer
    #   (activation function for first layer will usually be None.
    #   it can take values `sigmoid`, `tanh`, `relu`, `softmax`.)
    # cost_function: function to calculate error between predicated output and actual label/target
    #   it can take values `mean_squared`, `cross_entropy`.
    def __init__(self, num_layers, num_nodes, activation_functions, cost_function):
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.layers = []
        self.cost_function = cost_function

        for i in range(num_layers):
            if i != num_layers-1:
                layer_i = layer(num_nodes[i], num_nodes[i+1], activation_functions[i])
            else:
                layer_i = layer(num_nodes[i], 0, activation_functions[i])
            self.layers.append(layer_i)

    # batch_size: mini batch size for gradient descent
    # inputs: inputs to b e given to the network
    # labels: target values
    # num_epochs: number of epochs i.e. how many times the program should iterate over training
    # learning_rate: learning rate for the algorithm
    # filename: the name of the file that will finally store all variables after training
    #   (filename must have the extension .pkl)
    def train(self, batch_size, inputs, labels, num_epochs, learning_rate, filename):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        for j in range(num_epochs):
            print("== EPOCH: ", j, " ==")
            i = 0
            while i + batch_size != len(inputs):
                self.error = 0
                self.forward_pass(inputs[i:i+batch_size])
                self.calculate_error(labels[i:i+batch_size])
                self.back_pass(labels[i:i+batch_size])
                i += batch_size
            print("Error: ", self.error)
        dill.dump_session(filename)
    
    # inputs으로부터 각 layer의 activation_function에 따라 계산하고 다음 layer의 activations로 전파
    def forward_pass(self, inputs):
        self.layers[0].activations = inputs
        for i in range(self.num_layers - 1):
            temp = np.matmul(self.layers[i].activations, self.layers[i].weights_for_layer)
            act_func = self.layers[i+1].activation_function
            actvs = self.layers[i+1].activations
            if act_func == "sigmoid":
                actvs = self.sigmoid(temp)
            elif act_func == "softmax":
                actvs = self.softmax(temp)
            elif act_func == "relu":
                actvs = self.relu(temp)
            elif act_func == "tanh":
                actvs = self.tanh(temp)
            else:
                actvs = temp
    
    # sigmoide activation function is 'old-fashioned'
    def sigmoid(self, layer):
        return np.divide(1, np.add(1, np.exp(layer)))

    def softmax(self, layer):
        exp = np.exp(layer)
        if isinstance(layer[0], np.ndarray):
            return exp/np.sum(exp, axis=1, keepdims=True)
        else:
            return exp/np.sum(exp, keepdims=True)
    
    # rectified linear unit
    def relu(self, layer):
        layer[layer < 0] = 0
        return layer

    def tanh(self, layer):
        return np.tanh(layer)

    def calculate_error(self, labels):
        if len(labels[0]) != self.layers[self.num_layers-1].num_nodes_in_layer:
            print("Error: Label is not of the same shape as output layer")
            print("Label: ", len(labels), " : ", len(labels[0]))
            print("Out: ", len(self.layers[self.num_layers-1].activations), " : ", len(self.layers[self.num_layers-1].activations[0]))
            return
        
        if self.cost_function == "mean_squared":
            self.error = np.mean(
                np.divide(
                    np.square(
                        np.subtract(labels, self.layers[self.num_layers-1].activations)
                    ), 2
                )
            )
        elif self.cost_function == "cross_entropy":
            self.error = np.negative(
                np.sum(
                    np.multiply(labels, 
                        np.log(self.layers[self.num_layers-1].activations)
                    )
                )
            )

    # back propagation algorithm. it calculates gradient, multiplies it with a learning rate and subtracts the product from the existing weights
    def back_pass(self, labels):
        # if self.cost_function == "cross_entropy" and self.layers[self.num_layers-1].activation_function == "softmax":
        targets = labels
        i = self.num_layers-1
        actvs = self.layers[i].activations
        deltaw = np.matmul(np.asarray(self.layers[i-1].activations).T, np.multiply(actvs, np.multiply(1-actvs, targets-actvs)))
        new_weights = self.layers[i-1].weights_for_layer - self.learning_rate * deltaw
        for i in range(i-1, 0, -1):
            actvs = self.layers[i].activations
            weights = self.layers[i].weights_for_layer
            deltaw = np.matmul(np.asarray(self.layers[i-1].activations).T, np.multiply(actvs, np.multiply(1-actvs, np.sum(np.multiply(new_weights, weights),axis=1).T)))
            weights = new_weights
            new_weights = self.layers[i-1].weights_for_layer - self.learning_rate * deltaw
        self.layers[0].weights_for_layer = new_weights

    def predict(self, filename, input):
        dill.load_session(filename)
        self.batch_size = 1
        self.forward_pass(input)
        a = self.layers[self.num_layers-1].activations
        a[np.where(a==np.max(a))] = 1
        a[np.where(a!=np.max(a))] = 0
        return a
    
    def check_accuracy(self, filename, inputs, labels):
        dill.load_session(filename)
        self.batch_size = len(inputs)
        self.forward_pass(inputs)
        a = self.layers[self.num_layers-1].activations
        a[np.where(a==np.max(a))] = 1
        a[np.where(a!=np.max(a))] = 0
        total = 0
        correct = 0
        for i in range(len(a)):
            total += 1
            if np.equal(a[i], labels[i]).all():
                correct += 1
        print("Accuracy: ", correct*100/total)
    
    def load_model(self, filename):
        dill.load_session(filename)


class layer:
    def __init__(self, num_nodes_in_layer, num_nodes_in_next_layer, activation_function):
        self.num_nodes_in_layer = num_nodes_in_layer
        self.activation_function = activation_function
        self.activations = np.zeros([num_nodes_in_layer, 1])
        if num_nodes_in_next_layer != 0:
            self.weights_for_layer = np.random.normal(0, 0.001, size=(num_nodes_in_layer, num_nodes_in_next_layer))
        else:
            self.weights_for_layer = None
