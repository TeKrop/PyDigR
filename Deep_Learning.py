# -*- coding: utf8 -*-
# @author Stéphane BOSSARD
#Ne Fonctionne pas
#Il renvoie des nombre random à la fin 
#Il est possible que cela soit dû au redimensionnement de l'image lors du traitement
#En effet, lorque l'on affiche l'image rien n'apparait

#site pour ma partie
#http://www.pyimagesearch.com/2014/09/22/getting-started-deep-learning-python/
#https://pythonhosted.org/nolearn/dbn.html#example-mnist
#http://operationpixel.free.fr/traitementniveaudegris_inversion.php
#http://mldata.org/
#http://francoislouislaillier.developpez.com/Python/Tutoriel/InitiationNumpy/Tuto1/

#Les différentes installation à faire
#sudo pip install nolearn
#sudo apt-get install python-scipy
#sudo pip install -U scipy
#sudo apt-get install python-pip

# import the necessary packages
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from nolearn.dbn import DBN
from scipy import *
import Image
import numpy as np
import sys
import cv2
import ImageOps


def inversionCouleur(img):
	L,H = img.size
	im = Image.new("RGB",(L,H))
	for y in range(H):
		for x in range(L):
			p = img.getpixel((x,y))
			r=256-p[0]
			v=256-p[1]
			b=256-p[2]
			im.putpixel((x,y),(r,v,b))
	return im

# Recuperation de la base de donnees
print "[X] downloading data..."
dataset = datasets.fetch_mldata("MNIST Original")

# Séparation entre les donée de test et les donnée d'entrainement
(trainX, testX, trainY, testY) = train_test_split(
dataset.data / 255.0, dataset.target.astype("int0"), test_size = 0.33)

#Creation du reseau de croyance profonde avec 784 unité d'entrée qui corresponde au nombre de pixel par image
#On a 300 élément de sorties
#Le dix correspond au 10 valeur de chiffre possible
#learn_rates est le taux d'apprentissage
#learn_rate_decays 
#epoch est le nombre de noeud afin d'entrainer la base
#verbose indique si il y a un affichage pour le debug
dbn = DBN(
[trainX.shape[1], 300, 10],
learn_rates = 0.6,
learn_rate_decays = 0.9,
epochs = 1,
verbose = 1)
dbn.fit(trainX, trainY)

#Traitement de l'image pour pouvoir prédire le chiffre
img = Image.open("/home/auronmustang/PyDigR/img/5.jpg")
#ImgGray=ImageOps.grayscale(img)
ImgInver=inversionCouleur(img)
ImgGray=ImageOps.grayscale(ImgInver)
imgData=ImgGray.getdata()
tab = np.array(imgData)
tab = np.resize(tab,(1,784))

#Prediction du chiffre sur l'image donnée
pred = dbn.predict(tab)
image = (tab * 255).reshape((28, 28)).astype("uint16")
print "Predicted {0}".format(pred[0])
cv2.imshow("Digit", image)
cv2.waitKey(0)