from sklearn.externals import joblib
import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
import itertools

# Find and return descriptors for given image
def descriptors(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	sift = cv2.xfeatures2d.SIFT_create()
	kp, des = sift.detectAndCompute(gray,None)
	return des

# Training function, takes descriptor array and number for K-mean
def train(k,des_array):
	km = KMeans(n_clusters = k)
	km.fit(des_array)
	return km

# Retrieve files into training and test sets
def splitFilesInDirectory(dir):
	files = os.listdir(dir)
	files = files[:20]
	length = len(files)
	training_files = [(cv2.imread(dir + '/' + img),img, dir, True) for img in files[0::2] if img[0] != "."]
	test_files = [(cv2.imread(dir + '/' + img),img, dir, False) for img in files[1::2] if img[0] != "."]
	return (training_files, test_files)

# Populates table
def populateTable(categories,k):
	training_data = []
	test_data = []
	for category in categories:
		training_images, test_images = splitFilesInDirectory('categories/'+category)
		training_data = training_data + training_images
		test_data = test_data + test_images

	training_des_array = list(itertools.chain(*[descriptors(img[0]) for img in training_data]))
	kkm = train(k,training_des_array)
	codebook = kkm.cluster_centers_
	return codebook, [(img[1],img[2], img[3], kkm.predict(descriptors(img[0]))) for img in test_data+training_data]

# N of K, and desired cats
ks = [80, 90, 100, 190, 200, 500, 1000, 1500, 2000]
cats = ['accordion', 'bass', 'brontosaurus', 'pyramid', 'lobster', 'sunflower', 'hedgehog', 'ferry']

for k in ks:
	#Creates dir with handle
	handle = str(k)+'_'+str(len(cats))
	os.makedirs('data/'+handle+'/')


	# Defines and dumps table and codebook
	codebook, table = populateTable(cats,k)
	joblib.dump(codebook,'data/'+handle+'/codebook_'+handle+'.pkl')
	joblib.dump(table,'data/'+handle+'/table_'+handle+'.pkl')	
