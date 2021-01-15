#Training Working Memory task
import numpy as np
import sys
sys.path.insert(0, '../..') #This line adds '../..' to the path so we can import the net_framework python file
from RNN_model_GRAD import *
import tensorflow as tf
from tensorflow import keras
import json
from tqdm import tqdm
import os

num_iters = int(input("Enter number of training iterations: "))
num_nodes = int(input("Enter number of nodes: "))
#Defining Network
time_constant = 100 #ms
timestep = 5 #ms
noise_strength = .01
num_inputs = 3#4, for prompt

connectivity_matrix = np.ones((num_nodes, num_nodes))
weight_matrix = np.random.normal(0, 1/np.sqrt(num_nodes), (num_nodes, num_nodes))
for i in range(num_nodes):
    weight_matrix[i,i] = 0
    connectivity_matrix[i,i] = 0
weight_matrix = tf.Variable(weight_matrix)
connectivity_matrix = tf.constant(connectivity_matrix)

noise_weights = 1 * np.ones(num_nodes)
bias_weights = np.random.normal(0, 1/np.sqrt(num_inputs), num_nodes)
input_weights = np.random.normal(0, 1/np.sqrt(num_inputs), num_nodes)/2
#prompt_weights = np.random.normal(0, 1/np.sqrt(num_inputs), num_nodes)/2

input_weight_matrix = tf.constant(np.vstack((bias_weights, noise_weights, input_weights)))#prompt_weights

def rule_input(time):
    #No input for now
    return 0
'''
def prompt(time):
    #No input for now
    return 0

'''    
def bias(time):
    return 1
def noise(time):
    return np.sqrt(2 * time_constant/timestep) * noise_strength * np.random.normal(0, 1)


input_funcs = [bias, noise, rule_input]#, prompt]

init_activations = tf.constant(np.zeros((num_nodes, 1)))
output_weight_matrix = tf.constant(np.random.uniform(0, 1/np.sqrt(num_nodes), (1, num_nodes)))
        
network = RNN(weight_matrix, connectivity_matrix, init_activations, output_weight_matrix, time_constant = time_constant,
             timestep = timestep, activation_func = keras.activations.relu, output_nonlinearity = lambda x : x)

#Training Network
net_weight_history = {}
time = 10000
def gen_functions():
    wait_time = int(np.random.uniform(2000, 3000))
    chosen_vals = np.random.uniform(0, 2, 2)
    def rule_input(time):
        #running for 15 seconds = 15000ms
        if time < 1000:
            return np.random.normal(0, .01)
        elif time >= 1000 and time < 2000:
            return chosen_vals[0] + np.random.normal(0, .01)
        elif time >= 2000 and time < 2000 + wait_time:
        	return np.random.normal(0, .01)
        elif time >= 2000 + wait_time and time < 3000 + wait_time:
        	return chosen_vals[1] + np.random.normal(0, .01)
        else:
        	return np.random.normal(0, .01)
    '''
    def prompt(time):
    	if time < 3000 + wait_time:
    		return 0
    	else:
    		return 1
	'''
    def target_func(time):
        #running for 15 seconds = 15000ms
        if time < 3000 + wait_time:
            return 0.25
        else:
            return 0.5 * (chosen_vals[0] > chosen_vals[1]) + 1 * (chosen_vals[0] < chosen_vals[1])
    
    def error_mask_func(time):
        #Makes loss automatically 0 during switch for 100 ms.
        #Also used in next training section. 
        return 1
    return rule_input, target_func, error_mask_func #prompt

targets = []
inputs = []
error_masks = []
print('Preprocessing...', flush = True)
for iter in tqdm(range(num_iters * 50), leave = True, position = 0):
    rule_input, target_func, error_mask_func = gen_functions() #prompt
    input_funcs[2] = rule_input
    #input_funcs[3] = prompt
    targets.append(network.convert(time, [target_func]))
    inputs.append(network.convert(time, input_funcs))
    error_masks.append(network.convert(time, [error_mask_func]))
print('Training...', flush = True)
weight_history, losses = network.train(num_iters, targets, time, num_trials = 50, inputs = inputs,
              input_weight_matrix = input_weight_matrix, learning_rate = .001, epochs=50, error_mask = error_masks, save = 1)

net_weight_history['trained weights'] = np.asarray(weight_history).tolist()

net_weight_history['bias'] = bias_weights.tolist()
net_weight_history['noise weights'] = noise_weights.tolist()
net_weight_history['input weights'] = input_weights.tolist()
#net_weight_history['prompt weights'] = prompt_weights.tolist()
net_weight_history['connectivity matrix'] = np.asarray(connectivity_matrix).tolist()
net_weight_history['output weights'] = np.asarray(output_weight_matrix).tolist()

if not os.path.isdir(str(num_nodes) + '_nodes'):
    os.mkdir(str(num_nodes) + '_nodes')
with open(str(num_nodes) + '_nodes/weight_history.json', 'w') as f:
    json.dump(net_weight_history, f)