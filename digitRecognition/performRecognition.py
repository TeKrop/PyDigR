# -*- coding: utf8 -*-
# @author Stéphane BOSSARD
# Ne Fonctionne pas pour tous le cas il fonctionne avec plus ou moins debonne réponse pour photo_1 photo_2 photo_4
# Pour la plupart des autres cas, il donne 2 comme réponse
#avec photo_3 photo_5 photo_9 erreur ci dessous 
'''  
OpenCV Error: Assertion failed (ssize.area() > 0) in resize, file /build/buildd/opencv-2.3.1/modules/imgproc/src/imgwarp.cpp,
line 1428
File "performRecognition.py", line 88, in <module>
roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
cv2.error: /build/buildd/opencv-2.3.1/modules/imgproc/src/imgwarp.cpp:1428: error: (-215) ssize.area() > 0 in function resize
'''
#Cette erreur serait dû au fait que le programme doit réaliser des traitement à l'extérieur de l'image

#http://hanzratech.in/2015/02/24/handwritten-digit-recognition-using-opencv-sklearn-and-python.html

#installation
#pip install -U scikit-image

from sklearn.externals import joblib
from skimage.feature import hog
from sklearn import datasets
from sklearn.svm import LinearSVC
from collections import Counter
import cv2.cv as cv
import cv2
import numpy as np
import sys
import Image
import os.path

def LoadClassifier():
    dataset = datasets.fetch_mldata("MNIST Original")

    # Extrait de la base les donnees
    features = np.array(dataset.data, 'int16') 
    labels = np.array(dataset.target, 'int')

    # Extract the hog features
    list_hog_fd = []
    for feature in features:
        fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        list_hog_fd.append(fd)
    hog_features = np.array(list_hog_fd, 'float64')

    clf = LinearSVC()

    # On realise l'entrainement pour l'idendification
    clf.fit(hog_features, labels)

    # On enregistre le classifier
    joblib.dump(clf, "digits_cls.pkl", compress=3)


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

def find_bounding_box(img):
    # Initialisation des variables
    xmin = xmax = ymin = ymax = 0

    # On trouve les coordonnées de la bounding box
    (xmin, xmax) = (find_lim(img, cv.GetCol, 0, img.width), find_lim(img, cv.GetCol, img.width, 0))
    (ymin, ymax) = (find_lim(img, cv.GetRow, 0, img.height), find_lim(img, cv.GetRow, img.height, 0))

    # On retourne le résultat margé
    return (xmin - marge, xmax + marge, ymin - marge, ymax + marge)



# Chargement du classifier
if os.path.exists("./digits_cls.pkl") :
    print "Chargement du classifier"
    clf = joblib.load("digits_cls.pkl")
else :
    print "Creation du classifier et chargement"
    LoadClassifier()
    clf = joblib.load("digits_cls.pkl")

#On lit l'image qui est passer en paramètre
im = cv2.imread(sys.argv[1])

# On met l'image en niveau de gris et on applique un filtre de Gauss pour lisser l'image
im_gray = cv2.cvtColor(im , cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

# On réalise le seuillage de l'image
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

#On recherche les contours de l'image pour l'analyser 
ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#On recupere les rectangles entourant les nombres
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.
for rect in rects:
    # Dessine les rectangles sur l'image
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
    # Creation de la zone autour du nombre
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    # Redimensionne l'image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roitab = roi.tolist()
    roitab = np.asarray(roitab)
    #roifinal = cv2.dilate(roitab, (3, 3))
    # Calculate the HOG features
    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
    cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)


cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey()