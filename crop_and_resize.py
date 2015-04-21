# -*- coding: utf8 -*-
# Script de cropping et resizing automatique
# d'un chiffre dessiné sur une photo avec OpenCV
# @author Valentin PORCHET

# Importations
import cv2 # Open CV
import cv2.cv as cv
import numpy as np # Numpy
import sys # Paramètres système
import os.path # OS
import time # Temps

# Variables globales
marge = 10

# Permet de trouver les min et max de la bounding box du chiffre
# @param img : image seuillée inverse
# @param func : fonction d'obtention des données (en colonne ou en ligne)
# @param lim_min : limite minimale de la boucle de parcours
# @param lim_max : limite maximale de la boucle de parcours
def find_lim(img, func, lim_min, lim_max):
	lim = 0
	width = img.width
	height = img.height
	min_found = False

	# Pour toutes les colonnes (ou lignes)
	pixels = range(lim_min, lim_max) if lim_min < lim_max else list(reversed(range(lim_max, lim_min)))

	for i in pixels:
		data = func(img, i)  # On prend les données
		value = cv.Sum(data)  # On les somme

		# Si la somme supérieure à 0, il y a forcément
		# un pixel blanc, donc une partie du chiffre
		if (value[0] > 0):
			lim = i
			break

	return lim


# Permet de trouver la bounding box du chiffre
# @param img : image seuillée inverse contenant le chiffre
def find_bounding_box(img):
	# Initialisation des variables
	xmin = xmax = ymin = ymax = 0

	# On trouve les coordonnées de la bounding box
	(xmin, xmax) = (find_lim(img, cv.GetCol, 0, img.width), find_lim(img, cv.GetCol, img.width, 0))
	(ymin, ymax) = (find_lim(img, cv.GetRow, 0, img.height), find_lim(img, cv.GetRow, img.height, 0))

	# On retourne le résultat margé
	return (xmin - marge, xmax + marge, ymin - marge, ymax + marge)


# Permet de redimensionner l'image en gardant l'aspect ratio
# @param img : image
# @param new_width : nouvelle largeur
# @param new_height : nouvelle hauteur
def redim_picture(img, new_width, new_height):
	# Anciennes dimensions
	old_width = cv.GetSize(img)[0]
	old_height = cv.GetSize(img)[1]

	# Nouvelles dimensions
	new_width = int(sys.argv[2])
	new_height = int(sys.argv[3])

	# On définit la nouvelle image		
	final_picture = cv2.resize(np.array(img), (new_width, new_height))

	return final_picture

## FONCTION PRINCIPALE

# Si on a fourni le bon nombre d'arguments (2 ou 4) et si
# le fichier ciblé existe, alors on fait les manip
argc = len(sys.argv)
if (argc is 2 or argc is 4) and os.path.isfile(sys.argv[1]):

	print("Début à " + str(time.clock()) + " secondes")	

	# Chargement de l'image
	img = cv.LoadImage(sys.argv[1],0)

	# Seuillage
	img_seuil = img
	cv.Threshold(img, img_seuil, 127, 255, cv2.THRESH_BINARY_INV)

	# On trouve la bounding box
	bb = find_bounding_box(img_seuil)

	# On effectue le cropping
	final_picture = img[bb[2]:bb[3], bb[0]:bb[1]]

	# On réinverse les couleurs
	# Trouver mieux (genre la copie img_seuil = img à améliorer)
	cv.Threshold(img, img_seuil, 127, 255, cv2.THRESH_BINARY_INV)

	print("Crop réalisé en " + str(time.clock()) + " secondes")	

	# Si 4 arguments, on vérifie si on peut redimensionner
	if argc is 4:
		# Si les paramètres sont bien des nombres entiers positifs non nul
		if int(sys.argv[2]) > 0 and int(sys.argv[3]) > 0:
			# On utilise la fonction de redimensionnement
			final_picture = redim_picture(final_picture, int(sys.argv[2]), int(sys.argv[3]))
			print("Redimensionnement réalisé en " + str(time.clock()) + " secondes")
		else:
			print("Redimensionnement non effectué : dimensions données non entières")

	# Affichage		
	cv2.imshow("Image finale", np.asarray(final_picture))
	cv2.waitKey(0)
else:
	print("Utilisation : python crop.py [cheminImage] ([largeurFinale] [hauteurFinale])")
