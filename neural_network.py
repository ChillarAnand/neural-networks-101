import os
import sys
import pickle

import numpy as np
from scipy.special import expit
# from matplotlib import pyplot as plt


network_file = 'nn_movie.pkl'


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes
        self.lr = learning_rate
        self.verbose = False
        self.is_trained = False

        if os.path.exists('mnist.pkl'):
            with open(network_file, 'rb') as fh:
                network = pickle.load(fh)
            self.wih, self.who = network['wih'], network['who']
            self.is_trained = True
        else:
            self.wih = np.random.normal(
                0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes)
            )
            self.who = np.random.normal(
                0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes)
            )

        if len(sys.argv) > 1:
            print('Verbose output')
            self.verbose = True

        self.activation_func = lambda x: expit(x)

    def query(self, inputs):
        t_inputs = np.array(inputs, ndmin=1).T
        h_inputs = np.dot(self.wih, t_inputs)
        h_outputs = self.activation_func(h_inputs)
        o_inputs = np.dot(self.who, h_outputs)
        o_outputs = self.activation_func(o_inputs)
        if self.verbose:
            print("input")
            from pprint import pprint; pprint(inputs)
            print("transformed input")
            from pprint import pprint; pprint(t_inputs)
            print("network output")
            print(o_outputs)
        return o_outputs

    def train(self, inputs, targets):
        t_inputs = np.array(inputs, ndmin=2).T
        targets = np.array(targets, ndmin=2).T

        h_inputs = np.dot(self.wih, t_inputs)
        h_outputs = self.activation_func(h_inputs)

        o_inputs = np.dot(self.who, h_outputs)
        o_outputs = self.activation_func(o_inputs)

        o_errors = targets - o_outputs
        h_errors = np.dot(self.who.T, o_errors)


        if self.verbose:
            print("\ninput")
            from pprint import pprint; pprint(inputs)

            print("\ntransformed input")
            from pprint import pprint; pprint(t_inputs)

            print("\nWih")
            from pprint import pprint; pprint(self.wih)

            print("\nhidden layer inputs")
            from pprint import pprint; pprint(h_inputs)

            print("\nhidden layer outputs")
            from pprint import pprint; pprint(h_outputs)

            print("\nWho")
            from pprint import pprint; pprint(self.who)

            print("\noutput layer inputs")
            from pprint import pprint; pprint(o_inputs)

            print("\noutput layer outputs")
            from pprint import pprint; pprint(o_outputs)

            print("\noutput error")
            from pprint import pprint; pprint(o_errors)

            print("\nhidden layer errors")
            from pprint import pprint; pprint(h_errors)



        self.who += self.lr * np.dot((o_errors * o_outputs * (1 - o_outputs)), np.transpose(h_outputs))
        self.wih += self.lr * np.dot((h_errors * h_outputs * (1 - h_outputs)), np.transpose(t_inputs))

        if self.verbose:
            print("\nupdated Who")
            from pprint import pprint; pprint(self.who)
            print("\nupdated Wih")
            from pprint import pprint; pprint(self.wih)

        # print('===============================')
        # sys.exit()

# n = NueralNetwork(3, 3, 3, 0.3)
# print(n.query([0.4, -1.4, 1.1]))

# item = data[0].split(',')
# img_array = np.asfarray(item[1:]).reshape(28, 28)
# plt.imshow(img_array, cmap='Greys', interpolation='None')
# plt.show()

input_nodes = 5
hidden_nodes = 10
output_nodes = 2
learning_rate = 0.001


network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

train_samples_count = 40000
test_samples_count = 100

def get_sample():
    inputs = np.random.randint(low=0, high=10, size=5)
    prob = np.sum(inputs) / 50
    probs = (prob, 1 - prob)
    probs = (round(prob, 2), round(1 - prob, 2))
    labels = (round(prob), round(1 - prob))
    return (inputs, probs, labels)


scores = [np.random.randint(low=0, high=10, size=5) for i in range(train_samples_count)]
ratings = [(round(np.sum(i) / 50), round(1 - np.sum(i) / 50)) for i in scores]


print('Training network')
for i in range(train_samples_count):
    inputs, probs, labels = get_sample()
    network.train(inputs, labels)
    if i % 1000 == 0:
        print('sample: ' + str(i))
    # outputs = network.query(inputs)
    # print(outputs)
    # print('===============================')

scorecard = []

for i in range(test_samples_count):
    inputs, probs, labels = get_sample()
    outputs = network.query(inputs)
    o_labels = (round(outputs[0]), round(outputs[1]))

    print('inputs: ')
    # print(inputs, probs, labels)
    print(labels)
    print('network output')
    print(o_labels)

    pred_label = np.argmax(outputs)
    # print(inputs, labels, pred_label)
    print('=================')


network = {'wih': network.wih, 'who': network.who}

with open(network_file, 'wb') as fh:
    pickle.dump(network, fh)
