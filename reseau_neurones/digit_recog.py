
from neuron import Neuron
from PIL import Image
import os
import sys
import glob
import pickle

# neuron attributes
threshold  = 1
width      = 35
l_rate     = 0.005
err_margin = 0.1  # does nothing so far
a_func     = "step"
f_stretch  = 1

def main():
  
  print "=============="
  print "TRAINING PHASE"
  print "=============="
  neurons = list()
  try: # if first argument is provided, load as neuron data
    f = open(sys.argv[1], "r")
    neurons = pickle.load(f)
    print "Loaded " + sys.argv[1] + " as neuron data."
  except IndexError: # if no neuron data file is provided, ask to train anew
    yn = raw_input("No neuron data specified: train anew? (y/n) ")
    if yn.lower()[0] == "y":
      for x in range(10):
        neurons.append(Neuron(width, a_func, f_stretch, threshold, l_rate, err_margin))
      train(neurons)
      out_name = raw_input("Save name: ")
      out_file = open("data/" + out_name, "w")
      pickle.dump(neurons, out_file)
      print "Neuron data file saved in 'data/" + out_name + "'."
      print "Proceeding with current training."
    else:
      print "USAGE: to train anew : python digit_recog.py"
      print "       to load data  : python digit_recog.py data/your_data_file"
      print "Exiting."
      exit(1)
  except IOError: # if neuron data file is unreadable, print error msg and exit
    print "ERROR: neuron data file cannot be read."
    print "USAGE: to train anew : python digit_recog.py"
    print "       to load data  : python digit_recog.py data/your_data_file"
    exit(1)
    
  print "================="
  print "DIGIT RECOGNITION"
  print "================="
  # list compatible images
  compat = glob.glob("*.png")
  if len(compat) > 0:
    print "Compatible images found:"
    print "---"
    for png in compat:
      print png
    print "---"
  else:
    print "No compatible images found in current directory."
    print "---"
  
  # prompt to input image name
  while True:
    try:
      img_name = raw_input("Input image filename (Ctrl+C to exit): ")
    except KeyboardInterrupt:
      print
      print "Exiting."
      exit(0)
    # process image
    img = Image.open(img_name)
    
    counter = 0
    ans = None
    for x in range(len(neurons)):
      n = neurons[x]
      feed(img, n)
      n.activate()
      if n.get_output() == 1.0:
        print "Neuron %i is responding." %x
        ans = str(x)
        counter += 1
    
    if counter == 1: # if one neuron responded
      print img_name + " has been recognized as a " + ans + "."
    else: # if multiple or no neurons responded
      print img_name + " was unrecognizable."
  
  

def old_main():
  
  # neuron attributes
  threshold  = 1
  width      = 35
  l_rate     = 0.005
  err_margin = 0.1  # does nothing so far
  a_func     = "step"
  f_stretch  = 1
  
  # create list of neurons, one to be trained for each number
  neurons = list()
  
  for x in range(10):
    neurons.append(Neuron(width, a_func, f_stretch, threshold, l_rate, err_margin))
  
  # train or load from file if save file exists
  train(neurons)
  
  # trial
  test_img = Image.open(sys.argv[1])
  
  for x in range(len(neurons)):
    n = neurons[x]
    feed(test_img, n)
    n.activate()
    if n.get_output() == 1.0:
      print "NEURON %i IS RESPONDING" %x


def train(neurons):
  """
    Use data from training-sets directory to train.
  """
  for digit in range(len(neurons)):
    
    print "TRAINING FOR %i" %digit
    
    n = neurons[digit]
    
    ls1 = os.listdir("training-sets/")
    ls1 = sorted(ls1)
    
    
    errors  = 1
    counter = 0
    
    while errors > 0:
      errors = 0
      for i in ls1: # for every directory in training sets
        
        ### SET EXPECTED OUTPUT FOR CURRENT DIRECTORY
        try:
          exp_out = 0   # for other digits, training as not recognized (0)
          if int(i) == digit:
            exp_out = 1 # corresponding digit,  training as recognized (1)
        except ValueError:
          break # ignore directories that are not named as an integer
        
        ### FOR EACH IMAGE, FEED PIXEL SET AS INPUT AND USE exp_out TO CHECK
        dir1 = "training-sets/%s/" %i
        ls2 = os.listdir(dir1)
        ls2 = sorted(ls2)
        for j in ls2: # for every image
          
          img = Image.open(dir1 + j)
          feed(img, n)
          counter += 1
          errors += n.train_step(exp_out) # train with inputs
        # end of j-loop
      # end of i-loop
      print "Errors: %i" %errors
    # end of while loop
    
    
    
    print "Images processed: %i" %counter
    
  # end of for loop
  
# end of def train
  

def feed(img, n):
  """
    Sets the inputs of neuron according to pixel data provided of image
    PARAMS:
      img : Image to take pixel data from
      n   : neuron
  """
  pixels = img.load()
  width  = img.size[0]
  height = img.size[1]
  # distill pixel values into input nodes
  for w in range(width):
    for h in range(height):
      n.set_input(w * height + h, pixels[w,h][0]) # takes just the red value



if __name__ == "__main__":
  
  
  main()
