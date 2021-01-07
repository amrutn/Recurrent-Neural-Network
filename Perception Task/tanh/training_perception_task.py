#Training Simple Perceptual Decision Making task
import numpy as np
import sys
sys.path.insert(0, '../..') #This line adds '../..' to the path so we can import the net_framework python file
from RNN_model_GRAD import *
import tensorflow as tf
from tensorflow import keras
import json

net_num_iters = input("Enter num_iters for each training run separated by spaces: ")
net_num_iters = net_num_iters.split(' ')
net_num_iters = [int(num_iters) for num_iters in net_num_iters]
#Defining Network
num_nodes = 128
time_constant = 100 #ms
timestep = 10 #ms
noise_strength = .01
num_inputs = 3

connectivity_matrix = np.ones((num_nodes, num_nodes))
weight_matrix = np.random.normal(0, 1.2/np.sqrt(num_nodes), (num_nodes, num_nodes))
for i in range(num_nodes):
    weight_matrix[i,i] = 0
    connectivity_matrix[i,i] = 0
weight_matrix = tf.Variable(weight_matrix)
connectivity_matrix = tf.constant(connectivity_matrix)

noise_weights = 1 * np.ones(num_nodes)
bias_weights = np.random.normal(0, 1/np.sqrt(num_inputs), num_nodes)
input_weights = np.random.normal(0, 1/np.sqrt(num_inputs), num_nodes)

input_weight_matrix = tf.constant(np.vstack((bias_weights, noise_weights, input_weights)))

def rule_input(time):
    #No input for now
    return 0
    
def bias(time):
    return 1
def noise(time):
    return np.sqrt(2 * time_constant/timestep) * noise_strength * np.random.normal(0, 1)


input_funcs = [bias, noise, rule_input]

init_activations = tf.constant(np.zeros((num_nodes, 1)))
output_weight_matrix = tf.constant(np.random.uniform(0, 1/np.sqrt(num_nodes), (1, num_nodes)))
        
network = RNN(weight_matrix, connectivity_matrix, init_activations, output_weight_matrix, time_constant = time_constant,
             timestep = timestep, activation_func = keras.activations.tanh, output_nonlinearity = lambda x : x)

#Training Network
net_weight_history = {}


#Training low input, high output
num_iters = net_num_iters[0]
time = 5000 #ms

def rule_input(time):
    #running for 5 seconds = 5000ms, input =.2
    return .2 + np.random.normal(0, .02)
def target_func(time):
    #running for 5 seconds = 5000ms, out = .8.
    #Reverse of rule_input
    return .8

targets = network.convert(time, [target_func])
input_funcs[2] = rule_input
inputs = network.convert(time, input_funcs)

weight_history, losses = network.train(num_iters, targets, time, num_trials = 1, inputs = inputs,
              input_weight_matrix = input_weight_matrix, learning_rate = .001, save = 10)

net_weight_history['input=.2'] = np.asarray(weight_history).tolist()
#Training high input, low output
num_iters = net_num_iters[1]
time = 5000 #ms
def rule_input(time):
    #running for 5 seconds = 5000ms, input =.2
    return .8 + np.random.normal(0, .02)
def target_func(time):
    #running for 5 seconds = 5000ms, out = .8.
    #Reverse of rule_input
    return .2

targets = network.convert(time, [target_func])
input_funcs[2] = rule_input
inputs = network.convert(time, input_funcs)

weight_history, losses = network.train(num_iters, targets, time, num_trials = 1, inputs = inputs,
              input_weight_matrix = input_weight_matrix, learning_rate = .001, save = 10)

net_weight_history['input=.8'] = np.asarray(weight_history).tolist()
#Training on switching between low to high.
num_iters = net_num_iters[2]
time = 15000 #ms
def rule_input(time):
    #running for 15 seconds = 15000ms
    if time < 15000/2:
        return .2 + np.random.normal(0, .05)
    else:
        return .8 + np.random.normal(0, .05)
def target_func(time):
    #running for 15 seconds = 15000ms
    if time < 15000/2:
        return .8
    else:
        return .2
def error_mask_func(time):
    #Makes loss automatically 0 during switch for 100 ms.
    #Also used in next training section. 
    if time < 15000/2 + 50 and time > 15000/2 - 50:
        return 0
    else:
        return 1

targets = network.convert(time, [target_func])
input_funcs[2] = rule_input
inputs = network.convert(time, input_funcs)
error_mask = network.convert(time, [error_mask_func])
weight_history, losses = network.train(num_iters, targets, time, num_trials = 1, inputs = inputs,
              input_weight_matrix = input_weight_matrix, learning_rate = .001, error_mask = error_mask, save = 10)

net_weight_history['input low to high'] = np.asarray(weight_history).tolist()
#Training on switching between high to low.
num_iters = net_num_iters[3]
time = 15000 #ms
def rule_input(time):
    #running for 15 seconds = 15000ms
    if time < 15000/2:
        return .8 + np.random.normal(0, .05)
    else:
        return .2 + np.random.normal(0, .05)
def target_func(time):
    #running for 15 seconds = 15000ms
    if time < 15000/2:
        return .2
    else:
        return .8

targets = network.convert(time, [target_func])
input_funcs[2] = rule_input
inputs = network.convert(time, input_funcs)

weight_history, losses = network.train(num_iters, targets, time, num_trials = 1, inputs = inputs,
              input_weight_matrix = input_weight_matrix, learning_rate = .001, error_mask = error_mask, save = 10)

net_weight_history['input high to low'] = np.asarray(weight_history).tolist()

net_weight_history['bias'] = bias_weights.tolist()
net_weight_history['noise weights'] = noise_weights.tolist()
net_weight_history['input weights'] = input_weights.tolist()
net_weight_history['connectivity matrix'] = np.asarray(connectivity_matrix).tolist()
net_weight_history['output weights'] = np.asarray(output_weight_matrix).tolist()
with open('weight_history.json', 'w') as f:
	json.dump(net_weight_history, f)