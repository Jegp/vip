import numpy as np
import matplotlib.pyplot as plt
import ps_utils as ps

def pms(data):
	images, mask, light_vectors = ps.read_data_file(data)
	J = [images[:,:,i][np.where(mask)] for i in range(0,images.shape[2])]
	S = np.linalg.inv(light_vectors)
	m = np.dot(S,J)
	p = np.linalg.norm(m, axis=0)
	n = (1/p)*m
	n_list = [np.zeros((256,256)) for i in range(3)]
	for idx, ns in enumerate(n_list):
		ns[np.where(mask)] = n[idx]

	#return ps.unbiased_integrate(n_list[0],n_list[1],n_list[2], mask, order=2)
	return ps.simchony_integrate(n_list[0],n_list[1],n_list[2], mask)



#ps.display_depth_matplotlib(pms('Buddha.mat'))


images, mask, light_vectors = ps.read_data_file('Buddha.mat')

print(light_vectors)