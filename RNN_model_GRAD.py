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
	 	time_constant = 1, timestep = 0.2, activation_func = keras.activations.relu):
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
		#Basic tests to ensure correct input shapes.
		assert len(self.weight_matrix.shape) == 2
		assert self.weight_matrix.shape == self.connectivity_matrix.shape
		assert self.weight_matrix.shape[0] == self.weight_matrix.shape[1]
		assert len(self.output_weight_matrix.shape) == 2

		if internal != None:
			self.weights = internal
		if output != None:
			self.output_weight_matrix = output
		if connectivity != None:
			self.connectivity_matrix = connectivity
		where = tf.equal(self.connectivity_matrix, 0)
		self.mask = tf.constant(where.numpy() * self.weight_matrix.numpy())
		
	def convert(self, time, funcs = []):
		'''
		Converts a list of input functions into
		a matrix of different input values over time.
		 Makes it easier to create input and
		target matrices for simulation and training of the network. 

		Params
		------
		time : float
			Amount of time to simulate activity. The number of timesteps is given by
			time/self.timestep
		funcs : list of functions
			Returns value of functions for each time (not per timestep).
		
		Returns
		-------
		Matrix of input values where each column is an input value over time. 
		'''
		num_timesteps = int(time//self.timestep)
		result = 0
		if len(funcs) != 0:
			result = [[func(t * self.timestep) for func in funcs]\
			 for t in np.arange(0, num_timesteps, 1)]
			result = tf.constant(result)
			if result.dtype != tf.float32:
				result = tf.cast(result, 'float32')
		return result

	def simulate(self, time, inputs = tf.constant([[]]), input_weight_matrix = tf.constant([[]]),
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
		inputs : 2D Tensorflow Matrix Constant
			Matrix where each column is the value of a given input over time. 
		input_weight_matrix : 2D array tensorflow constant
			Weight matrix that represents how to add inputs to each 
			node. Has shape num_inputs x num_nodes. 
		'''

		if input_weight_matrix.dtype != tf.float32:
			input_weight_matrix = tf.cast(input_weight_matrix, 'float32')
		assert input_weight_matrix.shape[0] == inputs.shape[1]
		assert input_weight_matrix.shape[1] == self.num_nodes

		compiled_activations = []
		compiled_outputs = []
		num_timesteps = int(time//self.timestep)
		assert inputs.shape[0] == num_timesteps
		c = self.timestep/self.time_const

		if input_weight_matrix != tf.constant([[]]):
			add_inputs = tf.linalg.matmul(inputs, input_weight_matrix)

		for t in tqdm(range(num_timesteps), position = 0, leave = True, disable = disable_progress_bar):
			curr = self.activation

			add_inputs_t = 0
			if input_weight_matrix != tf.constant([[]]):	
				add_inputs_t = tf.transpose([add_inputs[t]])
				
			nxt = (1 - c) * curr + c * self.activation_func(\
				tf.linalg.matmul(self.weight_matrix, curr) + \
				add_inputs_t)
			self.activation = nxt
			outputs = tf.transpose(self.output_nonlinearity(tf.linalg.matmul(\
				self.output_weight_matrix, nxt)))[0]
			curr_activation = tf.transpose(self.activation)[0]
			compiled_outputs.append(outputs)
			compiled_activations.append(curr_activation)

		return compiled_outputs, compiled_activations

	def l2_loss_func(self, targets, time, num_trials = 1, regularizer = None,
	 inputs = tf.constant([[]]), input_weight_matrix = tf.constant([[]]),
	  error_mask = None):

		'''
		Computes loss function for the given weight matrix.

		Params
		------
		weight_matrix : tensorflow variable 2D weight matrix
			The weight matrix of the network
		targets : 2D Tensorflow array
			Matrix where each column is the value of a target output over time.
			Should have the same number of columns as the number of outputs. 
		time : float
			Amount of time (not timesteps) the training should simulate output. 
		num_trials : int
			number of trials to run the loss function. Is meant to get an average
			output if the network has stochastic inputs. 
		regularizer : function of the weight matrix (or None)
			regularization term in the loss function. 
		inputs : 2D Tensorflow array
			Matrix where each column is the value of a given input over time. 
		input_weight_matrix : 2D tensorflow constant matrix
			weight matrix for the inputs.
		error_mask : float or 2D tensorflow constant matrix
			Matrix of dimension num_timesteps x num_outputs. Defines
			how to weight network activity for each timestep at each
			output. (Same shape as the inputs and targets matrices).
		'''
		if regularizer == None:
			regularizer = lambda x : 0

		num_outputs = self.output_weight_matrix.numpy().shape[0]
		num_timesteps = int(time//self.timestep)

		assert targets[0].shape[1] == num_outputs

		summed = 0
		for trial in range(num_trials):

			curr_targets = targets[trial]
			curr_error_mask = error_mask[trial]
			curr_inputs = inputs[trial]
			if curr_targets.dtype != tf.float32:
				curr_targets = tf.cast(curr_targets, 'float32')
			self.reset_activations()
			simulated, _ = self.simulate(time, curr_inputs, input_weight_matrix, disable_progress_bar = True)
			l2 = 0
			for o in range(num_outputs):
				out_o = tf.transpose(simulated)[o]
				target_o = curr_targets[:, o]
				mask_o = curr_error_mask[:, o]
				l2 += tf.reduce_sum((out_o - target_o)**2 * mask_o)
			term = 1/(num_outputs * num_timesteps) * l2
			summed += tf.math.reduce_sum(term)
			
		return summed/num_trials + regularizer(self.weight_matrix)



	def train(self, num_iters, targets, time, learning_rate = 0.001,
	 num_trials = 1, regularizer = None, inputs = tf.constant([[]]),
	  input_weight_matrix = tf.constant([[]]), error_mask = None,
	   epochs = 10, save = 1):
		'''
		Trains the network using l2 loss. See other functions for the definitions of the parameters.
		For this function, instead of having one matrix as inputs/targets/error_mask, the user inputs
		a sequence of matrices. One for each training iteration. This allows for stochasticity in training.
		The parameter save tells us how often to save the weights/loss of the network. A value of 10 would
		result in the weights being saved every ten trials. 
		'''
		weight_history = []
		losses = []
		opt = keras.optimizers.Adam(learning_rate = learning_rate)
		if error_mask == None:
			num_outputs = self.output_weight_matrix.numpy().shape[0]
			num_timesteps = int(time//self.timestep)
			tmperror_mask = [tf.cast(tf.constant(np.ones((num_timesteps, num_outputs))), 'float32')]
		tmptargets = [targets]
		tmpinputs = [inputs]
		def loss():
			return self.l2_loss_func(tmptargets, time, num_trials, regularizer,\
	 			tmpinputs, input_weight_matrix, tmperror_mask)
		targets_len = len(targets)
		for iteration in tqdm(range(num_iters), position = 0, leave = True):

			#Randomly picking inputs/targets to use for this iteration
			pick_vals = np.random.choice(np.arange(0, targets_len, 1), num_trials, replace = False)
			tmptargets = []
			if len(inputs) != 0:
				tmpinputs = []
			if error_mask != None:
				tmperror_mask = []

			for val in pick_vals:
				tmptargets.append(targets[val])
				if len(inputs) != 0:
					tmpinputs.append(inputs[val])
				if error_mask != None:
					tmperror_mask.append(error_mask[val])
			
			opt.minimize(loss, [self.weight_matrix])
			loss_val = loss()
			if iteration % int(num_iters//epochs) == 0:
				print("The loss is: " + str(loss_val) + " at iteration " + str(iteration), flush = True)
			self.weight_matrix = tf.Variable(tf.identity(self.weight_matrix) * \
				self.connectivity_matrix + self.mask)
			if iteration % save == 0:
				weight_history.append(tf.identity(self.weight_matrix))
				losses.append(loss_val)
		return weight_history, losses
