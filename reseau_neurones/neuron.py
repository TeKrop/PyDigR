
import random
import math

quiet = True

class Neuron(object):
  """
    Last edited 02.02.2013 21:21
    
    Simulates a simple neural network.
    
    CLASS VARS :
    self.threshold  : when total input exceeds threshold, output is on
    self.inputs     : a list of inputs
    self.weights    : a list of weights
    self.output     : the float representation of the output
    self.l_rate     : learning rate
    self.err_margin : amount of allowable error
    self.a_func     : str activation function type (eg "step", "sigmoid")
    self.f_stretch  : activation function stretch (larger means less steep)
  """
  
  def __init__(self, width, a="step", s=1, t=1, l=0.1, e=0.1):
    self.threshold  = t
    self.inputs     = [0] * width
    self.weights    = [1] * width
    self.l_rate     = l
    self.err_margin = e
    self.a_func     = a
    self.f_stretch  = s
  
  def __str__(self):
    out  = "INPUTS\tWEIGHTS\n"
    for i in range(len(self.inputs)):
      out += str(self.inputs[i]) + "\t" + str(self.weights[i]) + "\n"
    out += "THRESHOLD  : " + str(self.threshold)  + "\n"
    out += "LEARN RATE : " + str(self.l_rate)     + "\n"
    out += "ERR MARGIN : " + str(self.err_margin) + "\n"
    return out
  
  def append(self, i, w):
    """
      Adds an input node and its corresponding weight.
    """
    self.inputs.append(i)
    self.weights.append(w)
    if len(inputs) != len(weights):
      print "Weight and input amount mismatch. Exiting."
      exit(1)
  
  def get_width(self): return len(self.inputs)
  
  def set_input(self, x, i): self.inputs[x] = i
  def get_input(self, x): return self.inputs[x]
  
  def set_weight(self, x, w): self.weights[x] = w
  def get_weight(self, x): return self.weights[x]
  
  def get_value(self, x): return self.weights[x] * self.inputs[x]
  
  def set_threshold(self, t): self.threshold = t
  def get_threshold(self): return self.threshold
  
  def set_rate(self, r): self.l_rate = r
  def get_rate(self): return self.l_rate
  
  def set_margin(self, em): self.err_margin = em
  def get_margin(self): return self.err_margin
  
  def activate(self):
    """
      Runs activation function designated in self.a_func and sets
      self.output accordingly.
    """
    i_sum = 0.0
    # perform summation operation of i_n * w_n
    for x in range(len(self.inputs)):
      i_sum += self.inputs[x] * self.weights[x]
    
    diff = i_sum - self.threshold
    diff /= self.f_stretch # account for f_stretch (only affects sigmoid)
    
    # using difference, determine output
    if self.a_func == "step":
      if diff > 0:
        self.output = 1.0
      else:
        self.output = 0.0
    elif self.a_func == "sigmoid":
      self.output = 1.0 / (1.0 + math.e**(-diff))
    else:
      print "Activation type incorrectly instantiated. Exiting."
      exit(1)
    if not quiet: print "Activation success."
    
    
  def get_output(self): return self.output
  def print_output(self): print "Output: %.1f" %self.output

  def train_step(self, exp_out):
    """
      Adjusts weights according to data in self.inputs
      PARAMS :
        exp_out : expected output
      RETURN :
        0 : no errors
        1 : error found
    """
    
    # perform activation
    self.activate()
    det_out = self.output # determined output
    if not quiet: print "My guess: %.1f" %det_out
    if not quiet: print "The expected output: %.1f" %exp_out
    
    # if output is not correct, calculate error
    error = 0
    error_rate = 0
    if det_out != exp_out:
      error = exp_out - det_out
      # adjust weights
      for x in range(len(self.inputs)):
        
        self.weights[x] += self.l_rate * error * self.inputs[x]
        if not quiet: print "New weight %i: %f" %(x+1,self.weights[x])
      
      self.threshold -= self.l_rate * error
      if not quiet: print "New threshold: %f" %self.threshold
      return 1 # return 1 for error
    else:
      if not quiet: print "Correct!"
      return 0 # return 0 for no errors

if __name__ == '__main__':
  n = Neuron(2) # width of two
