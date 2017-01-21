import ps_utils
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["QT_API"] = "pyqt"

(images, mask, light_vectors) = ps_utils.read_data_file("Beethoven.mat")

image_number = images.shape[2]
image_range = range(0, image_number)

def image_mask(image_number):
	array = [[], [], []]
	for x in range(0, 256):
		for y in range(0, 256):
			if mask[x][y] > 0:
				for n in range(0, image_number):
					array[n].append(images[x][y][n])
	return np.array(array)

# Albedo = dividing normal with square root of its own dot product
def albedo(matrix_norm):
	albedo_constant = np.sqrt(np.dot(matrix_norm, matrix_norm))
	if albedo_constant > 0:
		return matrix_norm / albedo_constant
	return 0

def images_from_mask(data):
	array = np.zeros([3, 256, 256])
	for i in image_range:
		counter = 0
		for x in range(0, 256):
			for y in range(0, 256):
				if mask[x][y] > 0:
					array[i][x][y] = data[i][counter]
					counter += 1
	return array

image_array = image_mask(3)
S_inverse = np.linalg.inv(light_vectors)
m = np.array([[pixel * S_inverse[i] for pixel in image_array[i]] for i in image_range])
albedo = np.array([[albedo(pixel) for pixel in arr] for arr in image_array])

albedo_images = images_from_mask(albedo)
plt.imsave('beethoven_albedo1.png', albedo_images[0], cmap='gray')
plt.imsave('beethoven_albedo2.png', albedo_images[1], cmap='gray')
plt.imsave('beethoven_albedo3.png', albedo_images[2], cmap='gray')

normal_field = np.zeros([256, 256, 3, 3])
counter = 0
for x in range(0, 256):
	for y in range(0, 256):
		if mask[x][y] > 0:
			normal_field[x][y] = np.array([m[0][counter], m[1][counter], m[2][counter]])
			counter += 1
n1 = np.array([[np.linalg.norm(pixel[0]) for pixel in row] for row in normal_field])
n2 = np.array([[np.linalg.norm(pixel[1]) for pixel in row] for row in normal_field])
n3 = np.array([[np.linalg.norm(pixel[2]) for pixel in row] for row in normal_field])

n1[n1==0] = np.nan
n2[n2==0] = np.nan
n3[n3==0] = np.nan

depth_field = ps_utils.simchony_integrate(n1, n2, n3, mask)
print(depth_field.shape)
ps_utils.display_depth_matplotlib(depth_field)

