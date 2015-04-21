# PyDigR
Python Digit Recognition

# Dépendances

* Python 2.7.x (incompatible Python 3.x)
* OpenCV2 (http://opencv.org/)
* Numpy (http://www.numpy.org/)

# Fichiers du projet

## crop_and_resize.py
    Utilisation : python crop_and_resize.py imageOrigine [imageFinale largeurFinale hauteurFinale]
    @param imageOrigine : chemin vers l'image à transformer
    @param imageOrigine : nom pour l'enregistrement de l'image finale transformée au format PNG (par défaut "image_finale")
    @param [largeurFinale hauteurFinale] : dimensions à rentrer pour le redimensionnement (optionnel)

Script python permettant de détecter une forme dans une image binaire et d'effectuer un rognage autour de celle-ci. Nous l'utilisons par exemple pour trouver un chiffre écrit noir sur blanc sur un tableau qui a été pris en photo.
(A voir si on implémente une méthode pour nettoyer le bruit environnant, avec méthode de connexité par exemple)
