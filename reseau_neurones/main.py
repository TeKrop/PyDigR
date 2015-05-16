# -*- coding: utf8 -*-
# Script de reconnaissance de digit
# Partie Réseau de neurones
# @author Valentin PORCHET

import sys # Pour utiliser le script de Alex Yang
import time # Pour le temps d'exécution

if len(sys.argv) > 3:
	generate = True if sys.argv[3] == "1" else False
else:
	generate = False

if len(sys.argv) > 2:
	training = True if sys.argv[2] == "1" else False
else:
	training = False

if len(sys.argv) > 1:
	# On va exécuter le crop and resize sur l'image donnée en paramètre
	# On met bien 5 et 7 pour correspondre à l'algo de Alex Yang
	sys.argv = ["crop_and_resize.py", sys.argv[1], "resized", "5", "7"]
	execfile("../crop_and_resize.py")

	# L'image est bien créée

	# Si jamais on veut faire le training du neurone, on le fait
	sys.argv = []
	if generate: # Si on veut générer les images avec le bruit
		execfile("digit_maker.py")
	if training: # Si on veut entraîner un nouveau neurone
		execfile("digit_recog.py")

	sys.argv = ["digit_recog.py", "data/neuron"]

	# On fait maintenant la reconnaissance
	execfile("digit_recog.py")
else:
	print("Utilisation : python main.py [cheminImage]")
