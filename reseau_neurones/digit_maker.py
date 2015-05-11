
from PIL import Image
import math
import random
import os

BLACK = (0,0,0)

noise_amt = 0.3

def add_noise(original, amt, num):
  """
    Takes a file and adds noise according to given amount (amt)
    PARAMS:
      original : filename of given file
      amt      : amount of noise to be added [0-1] (given as probability)
      num      : number to be appended to the end of the file name
  """
  
  img    = Image.open(original)
  width  = img.size[0]
  height = img.size[1]
  pixels = img.load()
  # must use digits to iterate through pixels
  for i in range(width):
    for j in range(height):
      x = random.random()
      print x
      if x < amt:
        y = random.randint(1,255)
        pixels[i,j] = (y, y, y, 255)
        print "Noise generated on pixel (%i, %i)" %(i,j)
      print pixels[i,j]
  
  out_filename = os.path.splitext(original)[0] + "_%i.png" %num
  img.save(out_filename, "PNG")
  print "Image saved as " + out_filename
  
  
  
if __name__ == "__main__":
  
  for x in range(0,10):
    for y in range(1, 31):
      add_noise("training-sets/%i/%i.png"%(x,x),noise_amt, y)
