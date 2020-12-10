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
	self.activation : 2D array
		tensorflow column vector of current activations for nodes in the network.
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

	'''

	def __init__(self, weight_matrix, connectivity_matrix, init_activations,
	 	output_weight_matrix, output_nonlinearity = keras.activations.sigmoid,
	 	time_constant = 1, timestep = 0.2, activation_func = keras.activations.softplus):
		'''
		Initializes an instance of the RNN class. 

		Params
		------
		See Attributes above
		'''

		#Basic tests to ensure correct input shapes.
		assert len(weight_matrix.shape) == 2
		assert weight_matrix.shape == connectivity_matrix.shape
		assert weight_matrix.shape[0] == weight_matrix.shape[1]
		assert len(init_activations.shape) == 2
		assert weight_matrix.shape[0] == init_activations.shape[0]
		assert len(output_weight_matrix.shape) == 2
		assert output_weight_matrix.shape[1] == init_activations.shape[0]

		#Ensuring correct dtype
		weight_matrix = tf.Variable(tf.cast(weight_matrix, 'float32'))
		connectivity_matrix = tf.constant(tf.cast(connectivity_matrix, 'float32'))
		output_weight_matrix = tf.constant(tf.cast(output_weight_matrix, 'float32'))
		init_activations = tf.constant(tf.cast(init_activations, 'float32'))
		#Setting attributes
		self.num_nodes = weight_matrix.shape[0]
		self.weight_matrix = weight_matrix
		self.connectivity_matrix = connectivity_matrix
		where = tf.equal(self.connectivity_matrix, 0)
		self.mask = tf.constant(where.numpy() * self.weight_matrix.numpy())
		self.output_weight_matrix = output_weight_matrix
		self.activation = init_activations
		self.activation_func = activation_func
		self.output_nonlinearity = output_nonlinearity
		self.time_const = time_constant
		self.timestep = timestep

	def reset_activations(self):
		self.activation =tf.zeros([self.num_nodes, 1], dtype = tf.dtypes.float32)

	def set_weights(self, internal = None, output = None, connectivity = None):
		'''
		Sets weights of the network
		'''
		if internal != None:
			self.weights = internal
		if output != None:
			self.output_weight_matrix = output
		if connectivity != None:
			self.connectivity_matrix = connectivity
		where = tf.equal(self.connectivity_matrix, 0)
		self.mask = tf.constant(where.numpy() * self.weight_matrix.numpy())
		#Basic tests to ensure correct input shapes.
		assert len(self.weight_matrix.shape) == 2
		assert self.weight_matrix.shape == self.connectivity_matrix.shape
		assert self.weight_matrix.shape[0] == self.weight_matrix.shape[1]
		assert len(self.output_weight_matrix.shape) == 2


	def simulate(self, time, input_funcs = [], input_weight_matrix = tf.constant([[]]),
	 disable_progress_bar = False):
		'''
		Simulates timesteps of the network given by time.
		time is equal to num_timesteps_simulated * self.timestep.
		returns all node activations over time and outputs
		over time. 

		Params
		------
		time : float
			Amount of time to simulate activity. The number of timesteps is given by
			time/self.timestep
		input_funcs : list of functions
			Returns value of inputs for each time (not per timestep).
		input_weight_matrix : 2D array tensorflow constant
			Weight matrix that represents how to add inputs to each 
			node.
		'''

		if input_weight_matrix.dtype != tf.float32:
			input_weight_matrix = tf.cast(input_weight_matrix, 'float32')
		compiled_activations = []
		compiled_outputs = []
		num_timesteps = int(time//self.timestep)
		c = self.timestep/self.time_const
		for t in tqdm(range(num_timesteps), position = 0, leave = True, disable = disable_progress_bar):
			curr = self.activation
			inputs = 0
			if len(input_funcs) != 0:
				inputs = [func(t * self.timestep) for func in input_funcs]
				inputs = tf.transpose(tf.constant([inputs]))
				if inputs.dtype != tf.float32:
					inputs = tf.cast(inputs, 'float32')
				add_inputs = tf.linalg.matmul(input_weight_matrix, inputs)
			else:
				add_inputs = 0
			nxt = (1 - c) * curr + c * self.activation_func(\
				tf.linalg.matmul(self.weight_matrix, curr) + \
				add_inputs)
			self.activation = nxt
			outputs = tf.transpose(self.output_nonlinearity(tf.linalg.matmul(\
				self.output_weight_matrix, nxt)))[0]
			curr_activation = tf.transpose(self.activation)[0]
			compiled_outputs.append(outputs)
			compiled_activations.append(curr_activation)

		return compiled_outputs, compiled_activations

	def l2_loss_func(self, target_functions, time, num_trials = 1, regularizer = None,
	 input_funcs = [], input_weight_matrix = tf.constant([[]]), error_mask = 1):

		'''
		Computes loss function for the given weight matrix.

		Params
		------
		weight_matrix : tensorflow variable 2D weight matrix
			The weight matrix of the network
		target_functions : list of functions
			returns the target outputs at given times (not timesteps). Should have
			same number of functions as the number of outputs. 
		time : float
			Amount of time (not timesteps) the training should simulate output. 
		num_trials : int
			number of trials to run the loss function. Is meant to get an average
			output if the network has stochastic inputs. 
		regularizer : function of the weight matrix (or None)
			regularization term in the loss function. 
		input_funcs : list of functions
			returns the input at given time (not timestep).
		input_weight_matrix : 2D tensorflow constant matrix
			weight matrix for the inputs.
		error_mask : float or 2D tensorflow constant matrix
			Matrix of dimension num_outputs x num_timesteps. Defines
			how to weight network activity for each timestep at each
			output.
		'''
		if regularizer == None:
			regularizer = lambda x : 0

		num_outputs = self.output_weight_matrix.numpy().shape[0]
		num_timesteps = int(time//self.timestep)
		if error_mask == 1:
			error_mask = np.ones((num_outputs, num_timesteps))
		assert len(target_functions) == num_outputs

		target = [[f(t * self.timestep) for t in range(num_timesteps)]\
		 for f in target_functions]

		target = tf.constant(target)
		if target.dtype != tf.float32:
			target = tf.cast(target, 'float32')
		summed = 0
		for trial in range(num_trials):
			simulated, _ = self.simulate(time, input_funcs, input_weight_matrix, disable_progress_bar = True)
			self.reset_activations()
			l2 = 0
			for o in range(num_outputs):
				out_o = tf.transpose(simulated)[o]
				target_o = target[o]
				mask_o = error_mask[o]
				l2 += tf.reduce_sum((out_o - target_o)**2 * mask_o)
			term = 1/(num_outputs * num_timesteps) * l2
			summed += tf.math.reduce_sum(term)
			
		return summed/num_trials + regularizer(self.weight_matrix)



	def train(self, num_iters, target_functions, time, learning_rate = 0.001, num_trials = 1, regularizer = None,
	 input_funcs = [], input_weight_matrix = tf.constant([[]]), error_mask = 1, epochs = 10):
		'''
		Trains the network using l2 loss. 
		'''
		weight_history = []
		opt = keras.optimizers.Adam(learning_rate = learning_rate)
		def loss():
			return self.l2_loss_func(target_functions, time, num_trials, regularizer,\
	 			input_funcs, input_weight_matrix, error_mask)

		for iteration in tqdm(range(num_iters), position = 0, leave = True):
			opt.minimize(loss, [self.weight_matrix])
			if iteration % int(num_iters//epochs) == 0:
				print("The loss is: " + str(loss()) + " at iteration " + str(iteration))
			self.weight_matrix = tf.Variable(tf.identity(self.weight_matrix) * \
				self.connectivity_matrix + self.mask)
			weight_history.append(tf.identity(self.weight_matrix))
		return weight_history
