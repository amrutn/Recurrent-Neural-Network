import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras

'''
This file is meant to implement a Recurrent Neural Network class 
that can be easily customized and analyzed.
We are training the network using the Adam optimizer.
'''



class RNN:
	'''
	A Class that represents a Recurrent Neural Network with arbitrary struture
	that is trained by implementing gradient descent using the Adam optimizer.
	The output of the network at each timestep is the weighted sum of each of 
	its input nodes with an optional nonlinearity. 

	Attributes
	----------
	self.weight_matrix : 2D array tensorflow variable
		Represents internal weights of the network.
	self.connectivity_matrix : 2D array tensorflow constant
		Array of ones and zeros. Only the internal weights 
		corresponding to ones in this matrix can be modified
		during training.
	self.mask : 2D array tensorflow constant
		Automatically generated "mask" matrix that is used to ensure
		certain weights are constant during training.
	self.output_weight_matrix : 2D array tensorflow constant
		Weight matrix that represents how to add internal node
		values to create each output. 
	self.input_weight_matrix : 2D array tensorflow constant
		Weight matrix that represents how to add inputs to each 
		node.
	self.activation : 1D array
		1D array of current activations for nodes in the network.
	self.activation_func : tensorflow activation function
		tensorflow function representing the nonlinearity on each
		internal node. 
	self.output_nonlinearity : tensorflow activation function
		tensorflow nonlinearity function applied to output. 
	self.time_const : float
		time constant of the decay of signal in eahc node of the
		neural network.
	self.timestep : float 
		timestep of the network.
	self.input_funcs : list of functions
		Returns value of inputs for each timestep. 
	'''

	def __init__(self, weight_matrix, connectivity_matrix, init_activations,
	 	output_weight_matrix, output_nonlinearity = keras.activations.sigmoid,
	 	input_funcs = [], input_weight_matrix = tf.constant([[]]),
	 	time_constant = 1, timestep = 0.2, activation_func = keras.activations.softplus):
		'''
		Initializes an instance of the RNN class. 

		Params
		------
		See Attributes above
		init_activations : 1D numpy array
			sets initial activations for each node of the network.
		'''

		#Basic tests to ensure correct input shapes.
		assert len(weight_matrix.shape) == 2
		assert weight_matrix.shape == connectivity_matrix.shape
		assert weight_matrix.shape[0] == weight_matrix.shape[1]
		assert len(init_activations.shape) == 1
		assert weight_matrix.shape[0] == init_activations.shape[0]
		assert len(output_weight_matrix.shape) == 2
		assert output_weight_matrix.shape[1] == init_activations.shape[0]
		assert len(input_weight_matrix.shape) == 2
		assert input_weight_matrix.shape[1] == len(input_funcs)
		if len(input_funcs) != 0:
			assert input_weight_matrix.shape[0] == init_activations.shape[0]

		#Setting attributes
		self.weight_matrix = weight_matrix
		self.connectivity_matrix = connectivity_matrix
		where = tf.equal(self.connectivity_matrix, 0)
		self.mask = tf.constant(where.numpy() * self.weight_matrix.numpy())
		self.output_weight_matrix = output_weight_matrix
		self.input_weight_matrix = input_weight_matrix
		self.activation = tf.transpose(tf.constant([init_activations]))
		self.activation_func = activation_func
		self.output_nonlinearity = output_nonlinearity
		self.time_const = time_constant
		self.timestep = timestep
		self.input_funcs = input_funcs


	def set_inputs(self, new_input_funcs):
		'''
		Changes input functions for each input node.
		Takes in a list of input functions.
		'''
		self.input_funcs = new_input_funcs

	def simulate(self, time):
		'''
		Simulates timesteps of the network given by time.
		time is equal to num_timesteps_simulated * self.timestep.
		returns all node activations over time and outputs
		over time. 
		'''
		compiled_activations = []
		compiled_outputs = []
		num_timesteps = int(time//self.timestep)
		c = self.timestep/self.time_const
		for t in tqdm(range(num_timesteps), position = 0, leave = True):
			curr = self.activation
			inputs = 0
			if len(self.input_funcs) != 0:
				inputs = [func(t) for func in self.input_funcs]
				inputs = tf.transpose(tf.constant([inputs]))
				inputs = tf.cast(inputs, 'float64')
				add_inputs = tf.linalg.matmul(self.input_weight_matrix, inputs)

			nxt = (1 - c) * curr + c * self.activation_func(\
				tf.linalg.matmul(self.weight_matrix, curr) + \
				add_inputs)
			self.activation = nxt
			outputs = self.output_nonlinearity(tf.linalg.matmul(\
				self.output_weight_matrix, nxt)).numpy().T[0]
			curr_activation = self.activation.numpy().T[0]
			compiled_outputs.append(outputs)
			compiled_activations.append(curr_activation)

		return np.asarray(compiled_outputs), np.asarray(compiled_activations)

