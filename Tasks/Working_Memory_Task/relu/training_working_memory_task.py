#Training Working Memory task
import numpy as np
import sys
sys.path.insert(0, '../../../') #This line adds '../../../' to the path so we can import the net_framework python file
from RNN_model_GRAD import *
import tensorflow as tf
from tensorflow import keras
import json
from tqdm import tqdm
import os

#Input1 is positively tuned input, input2 is negatively tuned input
num_iters = int(input("Enter number of training iterations: "))
num_nodes = int(input("Enter number of nodes: "))
#Defining Network
time_constant = 100 #ms
timestep = 10 #ms
noise_strength = .01
num_inputs = 4

connectivity_matrix = np.ones((num_nodes, num_nodes))
weight_matrix = np.random.normal(0, 1/np.sqrt(num_nodes), (num_nodes, num_nodes))
for i in range(num_nodes):
    weight_matrix[i,i] = 0
    connectivity_matrix[i,i] = 0
weight_matrix = tf.Variable(weight_matrix)
connectivity_matrix = tf.constant(connectivity_matrix)

noise_weights = 1 * np.ones(num_nodes)
bias_weights = np.random.normal(0, 1/np.sqrt(num_inputs), num_nodes)
input1_weights = np.random.normal(0, 1/np.sqrt(num_inputs), num_nodes)
input2_weights=input1_weights

input_weight_matrix = tf.constant(np.vstack((bias_weights, noise_weights, input1_weights, input2_weights)))

def input1(time):
    #No input for now
    return 0
def input2(time):
	#Negatively tuned input
	return 0
    
def bias(time):
    return 1
def noise(time):
    return np.sqrt(2 * time_constant/timestep) * noise_strength * np.random.normal(0, 1)


input_funcs = [bias, noise, input1, input2]

init_activations = tf.constant(np.zeros((num_nodes, 1)))
output_weight_matrix = tf.constant(np.random.uniform(0, 1/np.sqrt(num_nodes), (2, num_nodes)))
        
network = RNN(weight_matrix, connectivity_matrix, init_activations, output_weight_matrix, time_constant = time_constant,
             timestep = timestep, activation_func = keras.activations.relu, output_nonlinearity = lambda x : x)

#Training Network
net_weight_history = {}
time = 10000
def gen_functions():
    wait_time = int(np.random.uniform(2000, 3000))
    chosen_vals = np.random.uniform(0, 2, 2)

    def input1(time):
        if time >= 1000 and time < 2000:
            return chosen_vals[0] + np.random.normal(0, .01)
        elif time >= 2000 + wait_time and time < 3000 + wait_time:
            return chosen_vals[1] + np.random.normal(0, .01)
        else:
            return np.random.normal(0, .01)

    def input2(time):
        if time >= 1000 and time < 2000:
            return 2-chosen_vals[0] + np.random.normal(0, .01)
        elif time >= 2000 + wait_time and time < 3000 + wait_time:
            return 2-chosen_vals[1] + np.random.normal(0, .01)
        else:
        	return np.random.normal(0, .01)

    def target_func1(time):
        if time < 3000 + wait_time:
            return 0.1
        else:
            return 0.8 * (chosen_vals[0] > chosen_vals[1])

    def target_func2(time):
        if time < 3000 + wait_time:
            return 0.1
        else:
            return 0.8 * (chosen_vals[0] < chosen_vals[1])
    
    def error_mask_func1(time):
        #Makes loss automatically 0 before network decides
        #Also used in next training section. 
        return 1
    def error_mask_func2(time):
        #Makes loss automatically 0 before network decides
        #Also used in next training section. 
        return 1

    return input1,input2, target_func1,target_func2, error_mask_func1, error_mask_func2

targets = []
inputs = []
error_masks = []
print('Preprocessing...', flush = True)
for iter in tqdm(range(num_iters * 10), leave = True, position = 0):
    input1,input2, target_func1,target_func2, error_mask_func1, error_mask_func2 = gen_functions()
    input_funcs[2] = input1
    input_funcs[3] = input2
    targets.append(network.convert(time, [target_func1, target_func2]))
    inputs.append(network.convert(time, input_funcs))
    error_masks.append(network.convert(time, [error_mask_func1, error_mask_func2]))
print('Training...', flush = True)
weight_history, losses = network.train(num_iters, targets, time, num_trials = 10, inputs = inputs,
              input_weight_matrix = input_weight_matrix, learning_rate = .0005, epochs=50)#, error_mask = error_masks, save = 1)

net_weight_history['trained weights'] = np.asarray(weight_history).tolist()

net_weight_history['bias'] = bias_weights.tolist()
net_weight_history['noise weights'] = noise_weights.tolist()
net_weight_history['input1 weights'] = input1_weights.tolist()
net_weight_history['input2 weights'] = input2_weights.tolist()
net_weight_history['connectivity matrix'] = np.asarray(connectivity_matrix).tolist()
net_weight_history['output weights'] = np.asarray(output_weight_matrix).tolist()

if not os.path.isdir(str(num_nodes) + '_nodes'):
    os.mkdir(str(num_nodes) + '_nodes')
with open(str(num_nodes) + '_nodes/weight_history.json', 'w') as f:
    json.dump(net_weight_history, f)


