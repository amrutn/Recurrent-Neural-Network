import numpy as np
from tqdm import tqdm

'''
This file is meant to implement a Recurrent Neural Network class 
that can be easily customized and analyzed with time series data.
We are training the network useing FORCE learning with the Recursive Least 
Squares algorithm 
'''

class SingleNode:
	'''
	Class that represents a single node of the recurrent neural network. Sees and uses
	the attributes of its parent RNN. Can represent a regular node whose activation
	depends on its incoming edges and activation history. 
	'''
	def __init__(self, recurrent_net, name, init_activation, edge_indices, node_type,
	 activation_func = None, recurrent_weight = 0):
		'''
		Initializes a single node in the RNN. 

		Params
		------
		recurrent_net : Member of RNN class
			The RNN that this network is a part of.
		init_activation : float
			Initial activation of this node.
		name : int
			Name of the node. Set to its node index in the RNN
			class.
		init_activation : float
			Initial node activation.
		edge_indices : 1D numpy array
			Array of indices that correspond to the edges that
			point to this node in the RNN. Stored for easy
			feedforward computation for internal nodes.
		node_type : string or function
			"internal", "output", "input" or "feedback".
		activation_func : function
			The activation function of the node. Should have one input parameter.
		recurrent_weight : float
			How much of the previous raw activation of the neuron gets carried over 
			for each timestep. (Only applies to internal nodes.)
		'''

		self.activations = np.array([init_activation])
		self.timestep = 0
		self.recurrent_net = recurrent_net
		self.name = name
		self.activation_func = activation_func
		self.node_type = node_type
		self.edge_indices = edge_indices
		#Previous activation prior to nonlinearity (only for internalnodes)
		self.prev_raw_activation = init_activation
		self.recurrent_weight = recurrent_weight

	def get_activation(self, timestep):
		'''
		Returns the activation of the node at the given input timestep.
		Can only return values for timesteps that have already been simulated.
		'''
		assert timestep <= self.timestep, 'Node ' + str(self.name) + \
		 'has not been activated at timestep ' + timestep
		return self.activations[timestep]

	def activate(self):
		'''
		Adds the next activation to the stored activations array
		and returns the activation. Evaluates activation for self.timestep + 1.  
		'''
		if self.node_type == "output":
			activation = 0
			#Weights of the connections to this node
			connection_weights = self.recurrent_net.output_weights[self.name]
			for idx, node in enumerate(self.recurrent_net.internal_node_list):
				node_activ = node.get_activation(self.timestep)
				weight = connection_weights[idx]
				activation += weight * node_activ
				
		elif self.node_type == "input":
			activation = self.activation_func(self.timestep + 1)

		elif self.node_type == "internal":
			activation = self.recurrent_weight * self.prev_raw_activation
			#Adding up internal node presynaptic activations
			if self.edge_indices.size > 0:
				#Internal nodes that send signal to this node
				node_indices = self.recurrent_net.internal_edges[self.edge_indices][:, 0]
				presynaptic_nodes = [self.recurrent_net.internal_node_list[num] for num\
				in node_indices]
				#Weights of the connections to this node
				connection_weights = self.recurrent_net.internal_weights[self.edge_indices]
				for idx, node in enumerate(presynaptic_nodes):
					node_prev_activ = node.get_activation(self.timestep)
					weight = connection_weights[idx]
					activation += weight * node_prev_activ

			#Adding up input contributions
			input_nodes = self.recurrent_net.input_node_list
			if len(input_nodes) > 0:
				connection_weights = self.recurrent_net.input_weights[:, self.name]
				for idx, node in enumerate(input_nodes):
					node_prev_activ = node.get_activation(self.timestep)
					weight = connection_weights[idx]
					activation += weight * node_prev_activ

			#Adding up feedback contributions
			feedback_nodes = self.recurrent_net.feedback_node_list
			if len(feedback_nodes) > 0:
				connection_weights = self.recurrent_net.feedback_weights[:, self.name]
				for idx, node in enumerate(feedback_nodes):
					node_prev_activ = node.get_activation(self.timestep)
					weight = connection_weights[idx]
					activation += weight * node_prev_activ

			#Storing the raw activation prior to applying nonlinearity
			self.prev_raw_activation = activation
			activation = self.activation_func(activation)
		elif self.node_type == "feedback":
			activation = self.activation_func(self.recurrent_net)
		else:
			assert False, "Bad node type."

		self.activations = np.append(self.activations, activation)
		self.timestep += 1
		return activation



class RNN:
	'''
	A Class that represents a Recurrent Neural Network with arbitrary struture
	that is trained by implementing FORCE learning by using the Recursive 
	Least Squares algorithm. The output of the network at each timestep is the 
	weighted sum of each of its input nodes. 

	Attributes
	----------

	self.timestep : int
		Internal clock that keeps track of how many timesteps this network has been
		simulated for.
	self.num_(internal/output/input/feedback)_nodes : int
		Number of each type of nodes in the network
	self.(internal/output/input/feedback)_weights : Numpy Arrays
		Keeps tracks of weights in the network. See init params for 
		more info. 
	self.(internal/output/input/feedback)_node_list : list of SingleNode types
		A list of the nodes in the network. 
	self.weight_history : 2D numpy array
		Keeps tracks of a flattened version of the combined weights. Is updated
		every time the weights are updated. 
	'''

	def __init__(self, num_internal_nodes, init_activations, internal_edges,
	 	internal_weights, num_outputs, output_weights, input_funcs = [], 
		input_weights = np.array([[]]), recurrent_weight = 1, feedback_funcs = [], 
		feedback_weights = np.array([[]]), activation_func = np.tanh):
		'''
		Initializes an instance of the RNN class. 

		Params
		------
		num_internal_nodes : int
			The number of nodes in the network
		init_activations : 1D np array
			1D np array with length equal to num_internal_nodes. 
			Defines the initial activations of the network nodes. 
		internal_edges : 2D numpy array
			A Nx2 shaped numpy array where N is the number of edges in the network.
			Each row of the array represents a new edge pointing from node row[0] 
			to node row[1]. Indexing of the nodes starts from 0 and ends at num_internal_nodes - 1.
		internal_weights : 1D numpy array
			1D numpy array of length len(edges) that assigns a weight to each internal edge.
		num_outputs : int
			Number of outputs for the network
		output_weights : 2D numpy array
			2D array of dimensions num_nodes by num_outputs which assigns weights
			to an edge from each internal node. Each row is a set of weights for a
			given output node, so the number of rows is num_outputs and number
			of columns is num_internal_nodes.
		input_funcs : list of functions
			Defines what should be sent to each node as inputs at each timestep. 
			If there is no input, the list can be left empty. Otherwise, a list of
			functions with one parameter which can be called for different timesteps which
			will be used as an input function.
		input_weights : 2D numpy array of weights
			Same format as output_weights except for inputs.
		feedback_funcs : list of functions
			List of functions of self.
		feedback_weights : 2D numpy array
			Feedback weights for each feedback function. Same format as output_weights 
		recurrent_weight : 1D numpy array of length num_internal_nodes, or float
			Weights of the recurrent connections (prior to the nonlinearity). How much of the
			previous raw activation carries over to the next activation - see integrate and fire
			differential equation. Recurrent connections post-nonlinearity need to be added manually
			as weights. 
		activation_type : function
			Activation function for each node - should take in and return single float values.
		'''
		assert num_internal_nodes > 0,\
		 "Number of nodes must be larger than 0"
		assert np.amax(internal_edges) < num_internal_nodes,\
		 "Edge list has a node value out of range"
		assert np.all(np.equal(np.mod(internal_edges, 1), 0)),\
		 "All node values in edge list must integers from 0 to self.num_internal_nodes-1."
		assert init_activations.size == num_internal_nodes,\
		 "Initial activations list has wrong size."
		assert internal_weights.size == internal_edges.shape[0],\
		 'Internal weights and edges do not match.'
		if num_outputs > 0:
			assert output_weights.shape == (num_outputs, num_internal_nodes),\
			 'Output weights and number of outputs/nodes do not match'
		if len(input_funcs) > 0:
			assert input_weights.shape == (len(input_funcs), num_internal_nodes),\
			 'Input weights do not match number of inputs/nodes.'
		if len(feedback_funcs) > 0:
			assert feedback_weights.shape == (len(feedback_funcs), num_internal_nodes),\
			 'Output weights and number of feedbacks/nodes do not match'
		
		#Internal clock - should be in sync with the node interal clocks
		self.timestep = 0

		self.num_internal_nodes = num_internal_nodes
		self.num_output_nodes = num_outputs
		self.num_input_nodes = len(input_funcs)
		self.num_feedback_nodes = len(feedback_funcs)
		self.internal_edges = internal_edges.astype('int32')

		#Initializing weights in a numpy array that has an element for each edge.
		self.internal_weights = internal_weights
		self.output_weights = output_weights
		self.input_weights = input_weights
		self.feedback_weights = feedback_weights
		compiled_weights = np.concatenate((internal_weights, 
			output_weights.flatten(), input_weights.flatten(),
			 feedback_weights.flatten()))
		
		#List of internal nodes
		recurrent_weights = recurrent_weight * np.ones(self.num_internal_nodes)
		self.internal_node_list = []
		for num in range(self.num_internal_nodes):
			edge_indices = np.argwhere(self.internal_edges[:, 1] == num).T[0]
			edge_indices.astype("int32")			
			self.internal_node_list.append(SingleNode(self,
			 name = num, init_activation = init_activations[num],
			  edge_indices = edge_indices, node_type = "internal",
			   activation_func = activation_func,
			    recurrent_weight = recurrent_weights[num]))
		
		#List of input nodes
		self.input_node_list = []
		for num in range(self.num_input_nodes): 
			self.input_node_list.append(SingleNode(self,
			 name = num, init_activation = input_funcs[num](0),
			  edge_indices = [], node_type = "input",
			   activation_func = input_funcs[num]))

		#List of output nodes.
		self.output_node_list = []
		for num in range(self.num_output_nodes):
			self.output_node_list.append(SingleNode(self,
			 name = num, init_activation = 0,
			  edge_indices = [], node_type = "output"))
		
		#List of feedback nodes
		self.feedback_node_list = []
		for num in range(self.num_feedback_nodes):
			self.feedback_node_list.append(SingleNode(self,
			 name = num, init_activation = 0,
			  edge_indices = [], node_type = "feedback",
			   activation_func = feedback_funcs[num]))
			
		self.input_funcs = input_funcs
		self.feedback_funcs = feedback_funcs
		#Stores the weights of the network over time for training
		self.weight_history = [compiled_weights]

		#Placeholder matrices for FORCE training. Only need to train output and internal nodes since
		#all other nodes do not have trainable incoming edges.
		self.P_matrices_internal = [np.identity(1) for node in self.internal_node_list]
		self.P_matrices_output = [np.identity(1) for node in self.output_node_list]
	def get_internal_node_activations(self):
		'''
		Returns the complete history of activations for all internal nodes
		as a 2D numpy array where output[i] = activation history for node i.
		'''
		combined_activations = []
		for node in self.internal_node_list:
			combined_activations.append(node.activations)
		return np.array(combined_activations)
	def get_output_activations(self):
		'''
		Returns the complete history of activations for all output nodes
		as a 2D numpy array in same format as before.
		'''
		combined_activations = []
		for node in self.output_node_list:
			combined_activations.append(node.activations)
		return np.array(combined_activations)

	def get_input_signals(self):
		'''
		Returns the complete history of activations for all input nodes
		as a 2D numpy array in same format as before.
		'''
		combined_activations = []
		for node in self.input_node_list:
			combined_activations.append(node.activations)
		return np.array(combined_activations)

	def change_inputs(self, new_input_funcs):
		'''
		Changes input functions for each input node.
		Takes in a list of input functions. If an
		element of the list is None, does not change
		the input. 
		'''
		assert len(new_input_funcs) == self.num_input_nodes,\
		 "Input list length must be equal to the number of nodes."
		for idx, node in enumerate(self.input_node_list):
			new_func = new_input_funcs[idx]
			if new_func == None:
				continue
			node.activation_func = new_func

	def change_feedbacks(self, new_feedback_funcs):
		'''
		Changes feedback functions for each feedback node.
		Takes in a list of feedback functions. If an
		element of the list is None, does not change
		the feedback. 
		'''
		assert len(new_input_funcs) == self.num_input_nodes,\
		 "Input list length must be equal to the number of nodes."
		for idx, node in enumerate(self.feedback_node_list):
			new_func = new_feedback_funcs[idx]
			if new_func == None:
				continue
			node.activation_func = new_func

	def next(self):
		'''
		Simulates one timestep of the network. Does not return anything
		but all activations are stored.
		'''
		total_node_list = [*self.internal_node_list, \
		 *self.input_node_list, *self.output_node_list,\
		 *self.feedback_node_list]
		for node in total_node_list:
			node.activate()
		self.timestep += 1

	def simulate(self,time):
		'''
		Simulates multiple timesteps of the network given by time. Does not
		return anything but all activations are stored.
		'''
		for t in tqdm(range(time), position = 0, leave = True):
			self.next()

	
	def FORCE_train_next(self, function, num_iters, alpha = 1.0, timesteps_per_update = 3,
	 train_output = 0, train_weights = {'internal' : True, 'input' : True,
	  'output' : True, 'feedback' : True}):
		'''
		Trains all network weights using the Recursive Least Squares Algorithm. 

		Params
		------
		function : function
			Function to match output node to
		num_iters : int
			Number of completed training iterations
		alpha : float
			Inverse learning rate of the algorithm
		timesteps_per_update : int
			Number of timesteps to wait before updating weights
		train_output : int
			Which output node to train.
		train_weights : Dictionary
			Which weights to train from each set of edges. 
			Requires four keys: 
			'internal', 'input', 'output' and 'feedback'.
			Values can be True or False
			depending on whether or not those sets of weights
			will be trained. 
		''' 

		curr_time = self.timestep
		

		error = np.copy(self.get_output_activations()[:, -1])[train_output]\
		 - function(curr_time)

		if curr_time % timesteps_per_update == 0 and curr_time >= timesteps_per_update:
			
			#Training internal nodes
			for idx, node in enumerate(self.internal_node_list):
				presynaptic_nodes = []
				connection_weights = []
				if train_weights['internal']:
					edges_to_train = self.internal_edges[node.edge_indices][:, 0]
					nodes_to_train = [self.internal_node_list[num] for num in edges_to_train]
					presynaptic_nodes = nodes_to_train
					connection_weights = self.internal_weights[node.edge_indices]
				#Choosing input presynaptic nodes/weights
				if train_weights['input'] and self.num_input_nodes > 0:
					nodes_to_train = self.input_node_list
					for pre_node in nodes_to_train:
						presynaptic_nodes.append(pre_node)
					connection_weights = np.concatenate((connection_weights,\
					 self.input_weights[:, node.name]))

				#Choosing feedback presynaptic nodes/weights
				if train_weights['feedback'] and self.num_feedback_nodes > 0:
					nodes_to_train = self.feedback_node_list
					for pre_node in nodes_to_train:
						presynaptic_nodes.append(pre_node)
					connection_weights = np.concatenate((connection_weights,\
					 self.feedback_weights[:, node.name]))
				
				if presynaptic_nodes == []:
					continue
				node_activations = []
				for pre_node in presynaptic_nodes:
					node_prev_activ = node.get_activation(pre_node.timestep)
					node_activations.append(node_prev_activ)
				node_activations = np.array(node_activations)
				#Array of the inputs into each synapse
				node_activations = np.array([node_activations]).T
				if num_iters == 0:
					P = 1/alpha * np.identity(node_activations.size)
				else:
					P = self.P_matrices_internal[idx]
					P = P - (P @ node_activations @ node_activations.T @ P)/\
					(1 + node_activations.T @ P @ node_activations)
				self.P_matrices_internal[idx] = P
				
				connection_weights -= (error * P @ node_activations).T[0]

				assert not np.isnan(np.sum(connection_weights)), 'Training failed to converge'
				if train_weights['internal']:
					self.internal_weights[node.edge_indices] =\
					 connection_weights[0:node.edge_indices.size]
					if connection_weights.size > node.edge_indices.size:
						connection_weights = connection_weights[node.edge_indices.size:]
				if train_weights['input'] and self.num_input_nodes > 0:
					self.input_weights[:, node.name] = connection_weights[0:self.num_input_nodes]
					if connection_weights.size > self.num_input_nodes:
						connection_weights = connection_weights[self.num_input_nodes:]
				if train_weights['feedback'] and self.num_feedback_nodes > 0:
					self.feedback_weights[:, node.name] =\
					 connection_weights[0:self.num_feedback_nodes]


			#Training Output Nodes
			if train_weights['output'] and self.num_output_nodes > 0:
				for idx, node in enumerate(self.output_node_list):
 
					presynaptic_nodes = self.internal_node_list
					connection_weights = self.output_weights[idx]

					node_activations = []
					for node in presynaptic_nodes:
						node_prev_activ = node.get_activation(node.timestep)
						node_activations.append(node_prev_activ)
					node_activations = np.array(node_activations)
					#Array of the inputs into each synapse
					node_activations = np.array([node_activations]).T
					if num_iters == 0:
						P = 1/alpha * np.identity(node_activations.size)
					else:
						P = self.P_matrices_output[idx]
						P = P - (P @ node_activations @ node_activations.T @ P)/\
						(1 + node_activations.T @ P @ node_activations)
					self.P_matrices_output[idx] = P
					
					connection_weights -= (error * P @ node_activations).T[0]
					assert not np.isnan(np.sum(connection_weights)), 'Training failed to converge'
					self.output_weights[idx] = connection_weights
			
			compiled_weights = np.concatenate((self.internal_weights, 
				self.output_weights.flatten(), self.input_weights.flatten(),
				self.feedback_weights.flatten())) 

			self.weight_history.append(compiled_weights)
		self.next()

	def FORCE_train(self, time, function, alpha = 1.0, timesteps_per_update = 3, train_output = 0, 
		train_weights = {'internal' : True, 'input' : True, 'output' : True, 'feedback' : True}):
		'''
		Same as FORCE_train_next, but FORCE trains for multiple timesteps. time is the number of
		timesteps to continue training. All other inputs are the same. 
		'''

		for t in tqdm(range(time), position = 0, leave = True):
			self.FORCE_train_next(function, t, alpha, timesteps_per_update, train_output, train_weights)
