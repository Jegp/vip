from sklearn.externals import joblib
import numpy as np
#import matplotlib.pyplot as plt


# Load table from file
k_value = 100
file = str(k_value) + '_8'
table = joblib.load('data/'+file+'/table_'+file+'.pkl')

# Creates histogram
def createHist(table,k):
	return [(img[0], img[1], img[2], np.histogram(img[-1],k)[0]) for img in table]
table_hist = createHist(table,100)

# Retrives queries on test and train
# Returns statistical information in tuple
def traintestRetrieve(queries):
	train_table_hist = [t for t in table_hist if t[2]==True]
	test_table_hist = [t for t in table_hist if t[2]==False]
	
	train_rec_rank = []
	test_rec_rank = []

	for query in queries:
		query_cat = query[1]
		train_diff = [(np.linalg.norm(query[-1]-img[-1]), img[0], img[1]) for img in train_table_hist]
		test_diff = [(np.linalg.norm(query[-1]-img[-1]), img[0], img[1]) for img in test_table_hist]

		# Recripocal rank of train
		for idx, image in enumerate(sorted(train_diff)):
			if image[2] == query_cat:
				train_rec_rank.append(idx+1)
				break

		# Recripocal rank of test
		for idx, image in enumerate(sorted(test_diff)):
			if image[2] == query_cat:
				test_rec_rank.append(idx+1)
				break

	#Train rank for test and train
	train_rank = 0
	test_rank = 0
	for rank in train_rec_rank:
		if rank<4:
			train_rank += 1
	for rank in test_rec_rank:
		if rank<4:
			test_rank += 1
	train_rank = train_rank/len(queries)*100
	test_rank = test_rank/len(queries)*100

	return (train_rank, 1/np.mean(train_rec_rank)), (test_rank, 1/np.mean(test_rec_rank))


# Defines queries as test images
queries = [t for t in table_hist if t[2] == False]

# Retrieves information about test images from both train+test data
train_stats, test_stats = traintestRetrieve(queries)
print(train_stats,test_stats)


