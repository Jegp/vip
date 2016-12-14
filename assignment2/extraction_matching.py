import math 
from operator import itemgetter
from skimage.feature import corner_harris
import cv2
import skimage
from skimage import data, io, draw, color, transform, filters, feature
from skimage.io import imshow
from skimage.filters import rank
import scipy
from matplotlib import pyplot as plt
import numpy as np
from skimage.feature import blob_log
from skimage.color import rgb2gray


#Loads images and converts to grayscale
img1 = data.imread("Img001_diffuse_smallgray.png")
img1 = rgb2gray(img1)
img2 = data.imread("Img002_diffuse_smallgray.png")
img2 = rgb2gray(img2)
img3 = data.imread("Img009_diffuse_smallgray.png")
img3 = rgb2gray(img3)                


#Find and visualize blobs in all three images. Shows and saves results
fig,axes = plt.subplots(1, 3, figsize=(15,15))
blobs = [(img, blob_log(img, threshold=0.1, min_sigma=10, max_sigma=30)) for img in [img1, img2, img3]]    
for (img, blob) in blobs:
    ax = axes[0]
    axes = axes[1:]
    ax.set_title("")
    ax.imshow(img, interpolation='nearest',cmap='gray')
    for b in blob:
        y,x,r = b
        c = plt.Circle((x, y), r, linewidth=2, fill=False)
        ax.add_patch(c)
plt.savefig('blobs.png')
plt.show()

#Function that returns elements larger than zero
def minZero(x):
    return max(0, x)

#Defines size of patches. Can be altered as needed
radius = 10

#Creates lists of feature arrays in each image
feature_array1 = [(x, y, img1[minZero(x - radius):x + radius, minZero(y - radius):y + radius]) for x, y, sigma in blobs[0][1]]
feature_array2 = [(x, y, img2[minZero(x - radius):x + radius, minZero(y - radius):y + radius]) for x, y, sigma in blobs[1][1]]
feature_array3 = [(x, y, img3[minZero(x - radius):x + radius, minZero(y - radius):y + radius]) for x, y, sigma in blobs[2][1]]

#Function that normalizes and reshapes arrays that are too small
#The reshaping is done with mean padding
def normalize_array(a):
    x_diff = radius * 2 - a.shape[0]
    y_diff = radius * 2 - a.shape[1]
    if x_diff > 0 or y_diff > 0:
        #b = np.zeros((radius * 2, radius * 2))
        #b[int(x_diff / 2):int(radius * 2 - x_diff / 2), int(y_diff / 2):int(radius * 2 - y_diff / 2)] = a
        b =  np.lib.pad(a, [(0, x_diff), (0, y_diff)], mode = "mean")
        return b
    else:
        return a

#Finds the best feature pairs. The threshold can altered as needed
def find_features(features1, features2, threshold):
    features = []
    for feature1 in features1:
        current_features = []
        for feature2 in features2:
            n1 = normalize_array(feature1[2])
            n2 = normalize_array(feature2[2])
            c1 = (feature1[0], feature1[1])
            c2 = (feature2[0], feature2[1])
            difference = math.sqrt(((n1 - n2)**2).sum())
            current_features.append((c1, c2, difference))
        sorted_features = sorted(current_features, key = lambda t: t[2])
        if sorted_features[1][2] > 0 and sorted_features[0][2] / sorted_features[1][2] <= threshold:
            features.append(sorted_features[0])
    return features



#Plots matches between best feature pairs
def match(i1, f1, i2, f2, threshold,name):    
    features = find_features(f1, f2, threshold)

    img1_coordinates = np.array([t[0] for t in features])
    img2_coordinates = np.array([t[1] for t in features])
    matches = np.array([(i, i) for i in range(len(features))])

    f, ax = plt.subplots()
    plt.axis('off')
    feature.plot_matches(ax, i1, i2, img1_coordinates, img2_coordinates, matches)
    plt.savefig(name+'.png')

match(img1, feature_array1, img2, feature_array2, 0.8,'img1_img2')
match(img2, feature_array2, img3, feature_array3, 0.9,'img2_img3')
match(img1, feature_array1, img3, feature_array3, 0.9,'img1_img3')