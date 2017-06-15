from data_reader import *
from neural_net import *
from neural_net_impl import *
import sys
import random


def parseArgs(args):
  """Parses arguments vector, looking for switches of the form -key {optional value}.
  For example:
    parseArgs([ 'main.py', '-e', 20, '-r', 0.1, '-m', 'Simple' ]) = { '-e':20, '-r':5, '-t': 'simple' }
  """
  
  args_map = {}
  curkey = None
  for i in xrange(1, len(args)):
    if args[i][0] == '-':
      args_map[args[i]] = True
      curkey = args[i]
    else:
      assert curkey
      args_map[curkey] = args[i]
      curkey = None
  return args_map

def validateInput(args):
  args_map = parseArgs(args)
  assert '-e' in args_map, "A number of epochs should be specified with the flag -e (ex: -e 10)"
  assert '-r' in args_map, "A learning rate should be specified with the flag -r (ex: -r 0.1)"
  assert '-t' in args_map, "A network type should be provided. Options are: simple | hidden | custom"
  return(args_map)

def main():
  # Parsing command line arguments
  args_map = validateInput(sys.argv)
  epochs = int(args_map['-e'])
  rate = float(args_map['-r'])
  networkType = args_map['-t']

  # Load in the training data.
  images = DataReader.GetSortedImages('n_train.txt', 'p_train.txt')
  for image in images:
    assert len(image.pixels) == 36

  # Load the validation set.
  validation = DataReader.GetSortedImages('n_valid.txt', 'p_valid.txt')
  for image in validation:
    assert len(image.pixels) == 36
    
  test = DataReader.GetSortedImages('n_valid.txt', 'p_valid.txt')
  for image in test:
    assert len(image.pixels) == 36

  # Initializing network
  if networkType == 'simple':
    network = SimpleNetwork()
  if networkType == 'hidden':
    network = HiddenNetwork()
  if networkType == 'custom':
    network = CustomNetwork()

  # Hooks user-implemented functions to network
  network.FeedForwardFn = FeedForward
  network.TrainFn = Train

  # Initialize network weights
  network.InitializeWeights()
  
  #for i in range(len(network.network.weights)):
  #  print '%.8f' % network.network.weights[i].value
  
  #for i in range(len(network.network.hidden_nodes)):
  #	print '%d th node' %  i
  #	print network.network.hidden_nodes[i]
  #	for j in range(len(network.network.hidden_nodes[i].inputs)):
  #		print '%d th input' % j
  #		print network.network.hidden_nodes[i].inputs[j]
  
  #for i in range(len(network.network.outputs)):
  #	print '%d th output node' %  i
  #	print network.network.outputs[i]
  #	for j in range(len(network.network.outputs[i].inputs)):
  #		print '%d th output input' % j
  #		print network.network.outputs[i].inputs[j]
  		
  		
  # Displays information
  print '* * * * * * * * *'
  print 'Parameters => Epochs: %d, Learning Rate: %f' % (epochs, rate)
  print 'Type of network used: %s' % network.__class__.__name__
  print ('Input Nodes: %d, Hidden Nodes: %d, Output Nodes: %d' %
         (len(network.network.inputs), len(network.network.hidden_nodes),
          len(network.network.outputs)))
  print '* * * * * * * * *'
  # Train the network.
  output = network.Train(images, validation, test, rate, epochs)
  
  f = open('%s.%d.%f' % (network.__class__.__name__, epochs, rate), 'w')
  for i in range(len(output)):
  	f.write(str(output[i]) + '\n')

if __name__ == "__main__":
  main()
