from neural_net import NeuralNetwork, NetworkFramework
from neural_net import Node, Target, Input
import random

def FeedForward(network, input):
  """
  Arguments:
  ---------
  network : a NeuralNetwork instance
  input   : an Input instance

  Returns:
  --------
  Nothing

  Description:
  -----------
  This function propagates the inputs through the network. That is,
  it modifies the *raw_value* and *transformed_value* attributes of the
  nodes in the network, starting from the input nodes.

  Notes:
  -----
  The *input* arguments is an instance of Input, and contains just one
  attribute, *values*, which is a list of pixel values. The list is the
  same length as the number of input nodes in the network.

  i.e: len(input.values) == len(network.inputs)

  This is a distributed input encoding (see lecture notes 7 for more
  informations on encoding)

  In particular, you should initialize the input nodes using these input
  values:

  network.inputs[i].raw_value = input[i]
  """
  network.CheckComplete()
  # 1) Assign input values to input nodes
  for i in range(len(network.inputs)):
  	network.inputs[i].raw_value = input.values[i]
  	network.inputs[i].transformed_value = network.inputs[i].raw_value
    
	# 2) Propagates to hidden layer
  for i in range(len(network.hidden_nodes)):
    network.hidden_nodes[i].raw_value = network.ComputeRawValue(network.hidden_nodes[i])
    network.hidden_nodes[i].transformed_value = network.Sigmoid(network.hidden_nodes[i].raw_value)
    
	# 3) Propagates to the output layer
  for i in range(len(network.outputs)):
    network.outputs[i].raw_value = network.ComputeRawValue(network.outputs[i])
    network.outputs[i].transformed_value = network.Sigmoid(network.outputs[i].raw_value)

def Backprop(network, input, target, learning_rate):
  """
  Arguments:
  ---------
  network       : a NeuralNetwork instance
  input         : an Input instance
  target        : a Target instance
  learning_rate : the learning rate (a float)

  Returns:
  -------
  Nothing

  Description:
  -----------
  The function first propagates the inputs through the network
  using the Feedforward function, then backtracks and update the
  weights.

  Notes:
  ------
  The remarks made for *FeedForward* hold here too.

  The *target* argument is an instance of the class *Target* and
  has one attribute, *values*, which has the same length as the
  number of output nodes in the network.

  i.e: len(target.values) == len(network.outputs)

  In the distributed output encoding scenario, the target.values
  list has 10 elements.

  When computing the error of the output node, you should consider
  that for each output node, the target (that is, the true output)
  is target.values[i], and the predicted output is network.outputs[i].transformed_value.
  In particular, the error should be a function of:

  target.values[i] - network.outputs[i].transformed_value
  
  """
  network.CheckComplete()
  
  # 1) We first propagate the input through the network
  FeedForward(network, input)
  
  # 2) Then we compute the errors and update the weigths starting with the last layer
  for i in range(len(network.outputs)):
    error = target.values[i] - network.outputs[i].transformed_value
    network.outputs[i].delta = error * network.SigmoidPrime(network.outputs[i].raw_value)
    for j in range(len(network.outputs[i].weights)):
      network.outputs[i].weights[j].value += learning_rate * network.outputs[i].inputs[j].transformed_value * network.outputs[i].delta
    
  # 3) We now propagate the errors to the hidden layer, and update the weights there too
  for i in reversed(range(len(network.hidden_nodes))):
    error = 0.0
    for j in range(len(network.hidden_nodes[i].forward_neighbors)):
      error += network.hidden_nodes[i].forward_weights[j].value * network.hidden_nodes[i].forward_neighbors[j].delta
      network.hidden_nodes[i].delta = error * network.SigmoidPrime(network.hidden_nodes[i].raw_value)
    for j in range(len(network.hidden_nodes[i].weights)):
      network.hidden_nodes[i].weights[j].value += learning_rate * network.hidden_nodes[i].inputs[j].transformed_value * network.hidden_nodes[i].delta

def Train(network, inputs, targets, learning_rate, epochs):
  """
  Arguments:
  ---------
  network       : a NeuralNetwork instance
  inputs        : a list of Input instances
  targets       : a list of Target instances
  learning_rate : a learning_rate (a float)
  epochs        : a number of epochs (an integer)

  Returns:
  -------
  Nothing

  Description:
  -----------
  This function should train the network for a given number of epochs. That is,
  run the *Backprop* over the training set *epochs*-times
  """
  network.CheckComplete()
  
  for i in range(epochs):
    for j in range(len(inputs)):
      Backprop(network, inputs[j], targets[j], learning_rate)

class EncodedNetworkFramework(NetworkFramework):
  def __init__(self):
    super(EncodedNetworkFramework, self).__init__()
    
  def EncodeLabel(self, label):
    """
    Arguments:
    ---------
    label: a number between 0 and 9

    Returns:
    ---------
    a list of length 10 representing the distributed
    encoding of the output.

    Description:
    -----------
    Computes the distributed encoding of a given label.

    Example:
    -------
    0 => [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    3 => [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    Notes:
    ----
    Make sure that the elements of the encoding are floats.
    
    """
    
    List = [0.0 for x in range(10)]
    List[label] = 1.0
    target = Target()
    target.values = List
    return target

  def GetNetworkLabel(self):
    """
    Arguments:
    ---------
    Nothing

    Returns:
    -------
    the 'best matching' label corresponding to the current output encoding

    Description:
    -----------
    The function looks for the transformed_value of each output, then decides 
    which label to attribute to this list of outputs. The idea is to 'line up'
    the outputs, and consider that the label is the index of the output with the
    highest *transformed_value* attribute

    Example:
    -------
    # Imagine that we have:
    map(lambda node: node.transformed_value, self.network.outputs) => [0.2, 0.1, 0.01, 0.7, 0.23, 0.31, 0, 0, 0, 0.1, 0]
    # Then the returned value (i.e, the label) should be the index of the item 0.7,
    # which is 3
    """
    
    List = map(lambda node: node.transformed_value, self.network.outputs)
    return List.index(max(List))

  def Convert(self, image):
    """
    Arguments:
    ---------
    image: an Image instance

    Returns:
    -------
    an instance of Input

    Description:
    -----------
    The *image* arguments has 2 attributes: *label* which indicates
    the digit represented by the image, and *pixels* a matrix 14 x 14
    represented by a list (first list is the first row, second list the
    second row, ... ), containing numbers whose values are comprised
    between 0 and 256.0. The function transforms this into a unique list
    of 14 x 14 items, with normalized values (that is, the maximum possible
    value should be 1).
    "
    new_pixels = [None for i in range(196)]
	
    for i in range(14):
	    for j in range(14):
		    new_pixels[i*14+j] = image.pixels[i][j] / 256.0
    """

    input = Input()
    input.values = image.pixels
    return input		

  def InitializeWeights(self):
    """
    Arguments:
    ---------
    Nothing

    Returns:
    -------
    Nothing

    Description:
    -----------
    Initializes the weights with random values between [-0.01, 0.01].

    Hint:
    -----
    Consider the *random* module. You may use the the *weights* attribute
    of self.network.    
    """

    for weight in self.network.weights:
		weight.value = random.uniform(-.01, .01)

class SimpleNetwork(EncodedNetworkFramework):
  def __init__(self):
    """
    Arguments:
    ---------
    Nothing

    Returns:
    -------
    Nothing

    Description:
    -----------
    Initializes a simple network, with 196 input nodes,
    10 output nodes, and NO hidden nodes. Each input node
    should be connected to every output node.
    """
    
    super(SimpleNetwork, self).__init__()
    
    # 1) Adds an input node for each pixel.
    for i in range(36):
      self.network.AddNode(Node(), 1)
    # 2) Add an output node for each possible digit label.
    for i in range(1):
    	node = Node()
    	self.network.AddNode(node, 3)
    	for j in range(36):
    		node.AddInput(self.network.inputs[j],None,self.network)
    self.network.complete = True
    self.network.MarkAsComplete()

class HiddenNetwork(EncodedNetworkFramework):
  def __init__(self, number_of_hidden_nodes=10):
    """
    Arguments:
    ---------
    number_of_hidden_nodes : the number of hidden nodes to create (an integer)

    Returns:
    -------
    Nothing

    Description:
    -----------
    Initializes a network with a hidden layer. The network
    should have 196 input nodes, the specified number of
    hidden nodes, and 10 output nodes. The network should be,
    again, fully connected. That is, each input node is connected
    to every hidden node, and each hidden_node is connected to
    every output node.
    """
    
    super(HiddenNetwork, self).__init__()

    # 1) Adds an input node for each pixel
    for i in range(36):
		self.network.AddNode(Node(), 1)
    # 2) Adds the hidden layer
    for i in range(number_of_hidden_nodes):
    	node = Node()
    	self.network.AddNode(node, 2)
    	for j in range(36):
    		node.AddInput(self.network.inputs[j],None,self.network)
    # 3) Adds an output node for each possible digit label.
    for i in range(1):
    	node = Node()
    	self.network.AddNode(node, 3)
    	for j in range(number_of_hidden_nodes):
    		node.AddInput(self.network.hidden_nodes[j],None,self.network)
    self.network.complete = True
    self.network.MarkAsComplete()

class CustomNetwork(EncodedNetworkFramework):
  def __init__(self, number_of_hidden_nodes=10,number_of_layers=2):
    """
    Arguments:
    ---------
    number_of_hidden_nodes : the number of hidden nodes to create per layer (an integer)
    number_of_layers : the number of layers of hidden nodes

    Returns:
    --------
    Your pick

    Description:
    -----------
    Network with multiple hidden layers, all of the same size.  All layers fully connected.
    """
    super(CustomNetwork, self).__init__()
    
    # 1) Adds an input node for each pixel
    for i in range(36):
      self.network.AddNode(Node(), 1)
    # 2) Adds the first hidden layer
    for i in range(number_of_hidden_nodes):
    	node = Node()
    	self.network.AddNode(node, 2)
    	for j in range(36):
    		node.AddInput(self.network.inputs[j],None,self.network)
    # 2) Adds the next hidden layer(s)
    for k in range(number_of_layers - 1):
    	for i in range(number_of_hidden_nodes):
    		node = Node()
    		self.network.AddNode(node, 2)
    		for j in range(number_of_hidden_nodes):
    			node.AddInput(self.network.hidden_nodes[number_of_hidden_nodes*k + j],None,self.network)		
    # 3) Adds an output node for each possible digit label.
    for i in range(1):
    	node = Node()
    	self.network.AddNode(node, 3)
    	for j in range(number_of_hidden_nodes):
    		node.AddInput(self.network.hidden_nodes[number_of_hidden_nodes*(number_of_layers-1) + j],None,self.network)
    self.network.complete = True
    self.network.MarkAsComplete()
