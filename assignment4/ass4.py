import numpy as np
import matplotlib.pyplot as plt
import ps_utils as ps


images, mask, light_vectors = ps.read_data_file('Beethoven.mat')

# List of vectors with pixel values for pixels within the mask, for each image
J = [images[:,:,i][np.where(mask)] for i in range(0,3)]

# m = ???? (slides s. 39)
S = np.linalg.inv(light_vectors)
m = np.dot(S,J)

# p = Albedo, n = normal
p = np.linalg.norm(m, axis=0)

n = (1/p)*m
# = np.divide(m, p)


n1 = np.zeros((256,256))
n1[np.where(mask)] = n[0]
n2 = np.zeros((256,256))
n2[np.where(mask)] = n[1]
n3 = np.zeros((256,256))
n3[np.where(mask)] = n[2]

#plt.imshow(n3)
#plt.show()

#depth_map = ps.simchony_integrate(n1,n2,n3,mask)
depth_map = ps.unbiased_integrate(n1,n2,n3,mask, order=2)
ps.display_depth_matplotlib(depth_map)


#Create albedo image
albedo_image = np.zeros((256,256))
albedo_image[np.where(mask)] = p

#plt.imshow(matrix_albedo,cmap='gray')
#plt.show()
