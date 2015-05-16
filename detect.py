# -*- coding: Latin-1 -*-

# Importations
import cv2 # Open CV
import cv2.cv as cv
import numpy as np # Numpy
import sys # Paramètres système
import os.path # OS
import time # Temps
import crop_and_resize # Script du cropping

os.system("crop_and_resize.py img/2.jpg croped_5_7 5 7")
img = cv2.imread("croped_5_7.png")
