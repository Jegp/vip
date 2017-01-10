import cv2
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import itertools

des_array = np.array([])

def descriptors(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	sift = cv2.xfeatures2d.SIFT_create()
	kp, des = sift.detectAndCompute(gray,None)
	return des

def train(k,des_array):
	km = KMeans(n_clusters = k)
	km.fit(des_array)
	return km

# Retrieve files into training and test sets
def splitFilesInDirectory(dir):
	files = os.listdir(dir)
	length = len(files)
	training_files = [cv2.imread(dir + '/' + img) for img in files[length // 2:] if img[0] != "."]
	test_files = [cv2.imread(dir + '/' + img) for img in files[:length // 2] if img[0] != "."]
	return (training_files, test_files)

# Training data
(training_images, test_images) = splitFilesInDirectory('data')

training_des_array = list(itertools.chain(*[descriptors(img) for img in training_images]))

kkm = train(200,training_des_array)

codebook = kkm.cluster_centers_

training_labels = kkm.predict(descriptors(training_images[0]))
test_labels = kkm.predict(descriptors(test_images[0]))

print(test_labels)