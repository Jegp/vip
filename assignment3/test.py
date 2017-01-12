from sklearn.externals import joblib
import cv2
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import itertools
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

codebook = joblib.load('codebook.pkl')
table = joblib.load('table.pkl')

# Query

table_hist = [(img[0], img[1], img[2], np.histogram(img[-1],700)[0]) for img in table]

query = table_hist[-1]

closest = [(np.linalg.norm(query[-1]-img[-1]), img[0], img[1]) for img in table_hist]
print(sorted(closest)[2])