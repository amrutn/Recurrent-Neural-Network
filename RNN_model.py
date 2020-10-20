import numpy as np

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
	def __init__(self, recurrent_net, name, init_activation, edge_indices, node_type, activation_func):
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
			feedforward computation.
		node_type : string or function
			"regular", "output", or "input".
		activation_func : function
			The activation function of the node. Should have one input parameter.
		'''

		self.activations = np.array([init_activation])
		self.timestep = 0
		self.recurrent_net = recurrent_net
		self.name = name
		self.activation_func = activation_func
		self.node_type = node_type
		self.edge_indices = edge_indices

	def get_activation(self, timestep):
		'''
		Returns the activation of the node at the given input timestep.
		Can only return values for timesteps that have already been simulated.
		'''
		assert timestep <= self.timestep, 'Node ' + str(self.name) + 'has not been activated at timestep ' + timestep
		return self.activations[timestep]

	def activate(self):
		'''
		Adds the next activation to the stored activations array
		and returns the activation. Evaluates activation for self.timestep + 1.  
		'''
		if self.node_type == "output":
			activation = 0
			if self.edge_indices.size != 0:
				#Nodes that send input to this node
				node_connections = self.recurrent_net.edges[self.edge_indices][:, 0]
				#Weights of the connections to this node
				connection_weights = self.recurrent_net.weights[self.edge_indices]
				for idx, node_name in enumerate(node_connections):
					node_prev_activ = self.recurrent_net.node_list[node_name].get_activation(self.timestep)
					weight = connection_weights[idx]
					activation += weight * node_prev_activ

		elif self.node_type == "input":
			activation = self.activation_func(self.timestep + 1)

		elif self.node_type == "regular":
			activation = 0
			if self.edge_indices.size != 0:
				#Nodes that send input to this node
				node_connections = self.recurrent_net.edges[self.edge_indices][:, 0]
				#Weights of the connections to this node
				connection_weights = self.recurrent_net.weights[self.edge_indices]
				for idx, node_name in enumerate(node_connections):
					node_prev_activ = self.recurrent_net.node_list[node_name].get_activation(self.timestep)
					weight = connection_weights[idx]
					activation += weight * node_prev_activ
				activation = self.activation_func(activation)

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
	'''

	def __init__(self, num_nodes, edges, num_outputs = 1, recurrent_weight = 1, feedback =False, inputs = [], activation_type = "tanh", init_weights = ["randomUniform", [0.8, 1.2]], init_activations = ["randomUniform", [-5, 5]]):
		'''
		Initializes an instance of the RNN class. 

		Params
		------
		num_nodes : int
			The number of nodes in the network
		edges : 2D numpy array
			A Nx2 shaped numpy array where N is the number of edges in the network.
			Each row of the array represents a new edge pointing from node row[0] 
			to node row[1]. Indexing of the nodes starts from 0 and ends at num_nodes - 1.
		num_outputs : int
			Number of outputs for the network
		feedback : bool
			Whether or not to have feedback connections.
		recurrent_weight : 1D numpy array of length num_nodes, or float
			Weight of the recurrent connections.
		inputs : list of functions
			Defines what should be sent to each node as inputs at each timestep. Essentially
			acts as a new input node which is connected to every other node in the network
			with varying weights. Each element in the list can be considered a new input node.
			If there is no input, the list can be left empty. Otherwise, a list of
			functions with one parameter which can be called for different timesteps which
			will be used as an input function.
		activation_type : string (optional parameter)
			The type of activation function on each node. 
			The options are "none", "sigmoid", "tanh", "RELU"
		init_weights : list (optional parameter)
			A list of two elements, one string and one list of two floats or a single float, that specify 
			how to initialize weights for each node. For some floats z1, z2, options
			are ["randomUniform", [z1, z2]] which randomly initializes weights by sampling 
			from a uniform distribution between z1 and z2, ["randomNormal", [z1, z2]] 
			which randomly initializes weights by sampling from a Gaussian with mean z1 and 
			std z2, or ["constant", z] which initializes all weights to a constant
			value of z. Weights can be internal or external. Internal weights mediate
			node to node connections while the external weights mediate how the
			node activations combine to generate the output.
		init_activations : list (optional parameter)
			Determines the initial activations of the nodes in the network. Same format as
			init_weights.


		'''
		assert num_nodes > 0, "Number of nodes must be larger than 0"
		self.num_internal_nodes = num_nodes
		self.num_output_nodes = num_outputs
		self.num_input_nodes = len(inputs)
		assert np.amax(edges) < self.num_internal_nodes, "Edge list has a node value out of range"
		assert np.all(np.equal(np.mod(edges, 1), 0)), "All node values in edge list must integers from 0 to self.num_internal_nodes-1."
		#Adding recurrent connections.
		recurrent_edges = np.array([range(self.num_internal_nodes), range(self.num_internal_nodes)]).T
		self.edges = np.concatenate((edges, recurrent_edges), axis = 0)
		self.edges = self.edges.astype('int32')
		self.internal_edges = self.edges
		#Adding input connections (modulated by input weights)
		for i in range(self.num_input_nodes):
			input_edges = np.array([i + (self.num_internal_nodes) * np.ones(self.num_internal_nodes), range(self.num_internal_nodes)]).T
			self.edges = np.concatenate((self.edges, input_edges), axis = 0)
		self.edges = self.edges.astype('int32')
		self.input_edges = self.edges[self.internal_edges.shape[0]:]
		#Adding output connections (modulated by output weights)
		for i in range(self.num_output_nodes):
			output_edges = np.array([range(self.num_internal_nodes), 
				i + (self.num_internal_nodes + self.num_input_nodes) * np.ones(self.num_internal_nodes)]).T
			self.edges = np.concatenate((self.edges, output_edges), axis = 0)
		self.edges = self.edges.astype('int32')
		self.output_edges = self.edges[self.internal_edges.shape[0] + self.input_edges.shape[0]:]
		self.feedback = feedback
		#Adding feedback connections (modulated by feedback weights)
		if self.feedback:
			for i in range(self.num_output_nodes):
				feedback_edges = np.array([i + (self.num_internal_nodes + self.num_input_nodes)
				 * np.ones(self.num_internal_nodes), range(self.num_internal_nodes)]).T
				self.edges = np.concatenate((self.edges, feedback_edges), axis = 0)
			self.edges = self.edges.astype('int32')
			self.feedback_edges = self.edges[self.internal_edges.shape[0] + self.input_edges.shape[0] + self.output_edges.shape[0]:]

		#Defining the activation function
		if activation_type == "sigmoid":
			self.activation = lambda z : 1/(1 + np.exp(-z))
		elif activation_type == "tanh":
			self.activation = lambda z : (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))
		elif activation_type == "RELU":
			self.activation = lambda z : max(0, z)
		elif activation_type == "none":
			self.activation = lambda z : z
		else:
			assert False, "Invalid activation function"

		#Initializing internal and external weights in a numpy array that has
		#an element for each edge/node.
		z = init_weights[1]
		if init_weights[0] == "randomUniform":
			internal_weights = np.random.uniform(low = z[0], high = z[1], size = self.internal_edges.shape[0])
			#Setting weights of recurrent connections
			internal_weights[edges.shape[0]:] = recurrent_weight * np.ones(self.num_internal_nodes)
			output_weights = np.random.uniform(low = z[0], high = z[1], 
				size = self.num_internal_nodes * self.num_output_nodes)
			input_weights = np.random.uniform(low = z[0], high = z[1], 
				size = self.num_internal_nodes * self.num_input_nodes)
			if self.feedback:
				feedback_weights = np.random.uniform(low = z[0], high = z[1], 
					size = self.num_internal_nodes * self.num_output_nodes)
		elif init_weights[0] == "randomNormal":
			internal_weights = np.random.normal(loc = z[0], scale = z[1], size = self.internal_edges.shape[0])
			#Setting weights of recurrent connections
			internal_weights[edges.shape[0]:] = recurrent_weight * np.ones(self.num_internal_nodes)
			output_weights = np.random.normal(loc = z[0], scale = z[1], 
				size = self.num_internal_nodes * self.num_output_nodes)
			input_weights = np.random.normal(loc = z[0], scale = z[1], 
				size = self.num_internal_nodes * self.num_input_nodes)
			if self.feedback:
				feedback_weights = np.random.normal(loc = z[0], scale = z[1], 
					size = self.num_internal_nodes * self.num_output_nodes)
		elif init_weights[0] == "constant":
			internal_weights = z * np.ones(self.internal_edges.shape[0])
			#Setting weights of recurrent connections
			internal_weights[edges.shape[0]:] = recurrent_weight * np.ones(self.num_internal_nodes)
			output_weights = z * np.ones(self.num_internal_nodes * self.num_output_nodes)
			input_weights = z * np.ones(self.num_internal_nodes * self.num_input_nodes)
			if self.feedback:
				feedback_weights = z * np.ones(size = self.num_internal_nodes * self.num_output_nodes)
		else:
			assert False, "Invalid Weight Initialization"

		self.weights = np.concatenate((internal_weights, input_weights, output_weights))
		if self.feedback:
			self.weights = np.concatenate((self.weights, feedback_weights))
		#Initializing activation for each node in a numpy array.
		z = init_activations[1]
		if init_activations[0] == "randomUniform":
			activations = np.random.uniform(low = z[0], high = z[1], size = self.num_internal_nodes)
		elif init_activations[0] == "randomNormal":
			activations = np.random.normal(loc = z[0], scale = z[1], size = self.num_internal_nodes)
		elif init_activations[0] == "constant":
			activations = z * np.ones(self.num_internal_nodes)
		else:
			assert False, "Invalid Activation Initialization"


		#List of nodes
		self.node_list = []
		for num in range(self.num_internal_nodes):
			edge_indices = np.argwhere(self.edges[:, 1] == num).T[0]
			self.node_list.append(SingleNode(self, num, activations[num], edge_indices, node_type = "regular", activation_func = self.activation))
		#Adding input nodes
		for num in range(self.num_input_nodes):
			edge_indices = []
			name = num + self.num_internal_nodes
			self.node_list.append(SingleNode(self, name, inputs[num](0), edge_indices, node_type = "input", activation_func = inputs[num]))

		#Adding output/feedback nodes.
		for num in range(self.num_output_nodes):
			name = num + self.num_internal_nodes + self.num_input_nodes
			edge_indices = np.argwhere(self.edges[:, 1] == name).T[0]
			self.node_list.append(SingleNode(self, name, 0, edge_indices, node_type = "output", activation_func = None))


	def get_internal_node_list(self):
		'''
		Returns array of internal node pobjects.
		'''
		return self.node_list[: self.num_internal_nodes]
	def get_output_node_list(self):
		'''
		returns array of output node objects
		'''
		return self.node_list[self.num_internal_nodes + self.num_input_nodes:]

	def get_input_node_list(self):
		'''
		returns array of input node objects
		'''
		return self.node_list[self.num_internal_nodes:self.num_internal_nodes + self.num_input_nodes]
	def get_internal_node_activations(self):
		'''
		Returns the complete history of activations for all internal nodes
		as a 2D numpy array where output[i] = activation history for node i.
		'''
		combined_activations = []
		for node in self.get_internal_node_list():
			combined_activations.append(node.activations)
		return np.array(combined_activations)
	def get_output_activations(self):
		'''
		Returns the complete history of activations for all output nodes
		as a 2D numpy array in same format as before.
		'''
		combined_activations = []
		for node in self.get_output_node_list():
			combined_activations.append(node.activations)
		return np.array(combined_activations)

	def get_input_signals(self):
		'''
		Returns the complete history of activations for all input nodes
		as a 2D numpy array in same format as before.
		'''
		combined_activations = []
		for node in self.get_input_node_list():
			combined_activations.append(node.activations)
		return np.array(combined_activations)

	def next(self):
		'''
		Simulates one timestep of the network. Does not return anything
		but all activations are stored.
		'''
		for node in self.node_list:
			node.activate()
	def simulate(self,time):
		'''
		Simulates multiple timesteps of the network given by time. Does not
		return anything but all activations are stored.
		'''
		for t in range(time):
			self.next()

	'''
	def FORCE_train_next(self):
		#Which weights to train

	def FORCE_train(self,time,):
		#Which weights to train


	'''
