
# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: Simplified BSD

# Standard scientific Python imports
import pylab as pl
import numpy as np
import os
import cv2
import glob

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
import skimage.io


os.system("python crop_and_resize.py img/3.png samples/3 8 8")



def load_unknowndata(filenames):
    training = {'images': [], 'targets': [], 'data' : [], 'name' : []}

    for index, filename in enumerate(filenames):
        image = skimage.io.imread(filename)
        training['targets'].append(-1) # Target = -1: unkown
        training['images'].append(image)
        training['name'].append(filename)
        training['data'].append(image.flatten().tolist())

    ## To apply an classifier on this data, we need to flatten the image, to
    ## turn the data in a (samples, feature) matrix:
    n_samples = len(training['images'])
    training['images'] = np.array(training['images'])
    training['targets'] = np.array(training['targets'])
    training['data'] = np.array(training['data'])
    return training





# The digits dataset
digits = datasets.load_digits()

# The data that we are interested in is made of 8x8 images of digits,
# let's have a look at the first 3 images, stored in the `images`
# attribute of the dataset. If we were working from image files, we
# could load them using pylab.imread. For these images know which
# digit they represent: it is given in the 'target' of the dataset.
for index, (image, label) in enumerate(zip(digits.images, digits.target)[:4]):
    pl.subplot(2, 4, index + 1)
    pl.axis('off')
    pl.imshow(image, cmap=pl.cm.gray_r, interpolation='nearest')
    pl.title('Training: %i' % label)

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
classifier.fit(data[:n_samples / 2], digits.target[:n_samples / 2])

# Now predict the value of the digit on the second half:
expected = digits.target[n_samples / 2:]
predicted = classifier.predict(data[n_samples / 2:])

print "Classification report for classifier %s:\n%s\n" % (
    classifier, metrics.classification_report(expected, predicted))
print "Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted)

for index, (image, prediction) in enumerate(
    zip(digits.images[n_samples / 2:], predicted)[:4]):
    pl.subplot(2, 4, index + 5)
    pl.axis('off')
    pl.imshow(image, cmap=pl.cm.gray_r, interpolation='nearest')
    pl.title('Prediction: %i' % prediction)

pl.show()