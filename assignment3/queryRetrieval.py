from sklearn.externals import joblib
import numpy as np

# Load table from file
file = '60_8'
table = joblib.load('data/'+file+'/table_'+file+'.pkl')
k = int(file[:-2])
#k = 100


# Creates histogram
def createHist(table,k):
	return [(img[0], img[1], img[2], np.histogram(img[-1],k)[0]) for img in table]



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
		train_diff = sorted(train_diff)
		test_diff = sorted(test_diff)[1:]

		# Recripocal rank of train
		for idx, image in enumerate(train_diff):
			if image[2] == query_cat:
				train_rec_rank.append(idx+1)
				break

		# Recripocal rank of test
		for idx, image in enumerate(test_diff):
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


#Root files = ['80_8', '120_8', '150_8', '100_8', '250_8', '500_8', '1000_8', '1500_8', '2000_8']
#files = ['100_8']
#X files =  ['100_8', '200_8', '300_8', '400_8', '500_8', '1000_8', '1500_8', '1750_8']
#Z files = ['10_7', '20_7', '30_7', '40_7', '50_7', '60_7', '70_7', '80_7', '90_7', '100_7', '200_7', '300_7', '400_7', '500_7', '1000_7', '1500_7', '2000_7']
#files = ['100_8', '190_8', '200_8', '500_8', '1000_8', '1500_8', '2000_8']
files = ['80_8']
#Loads all files and print stats for each file/k

for file in files:
	#Load table, define k, create histograms
	table = joblib.load('data/'+file+'/table_'+file+'.pkl')
	k = int(file[:-2])
	table_hist = createHist(table,k)
	
	# Defines queries as test images
	queries = [t for t in table_hist if t[2] == False]

	# Retrieves information about test images from both train+test data
	print(k,':')
	train_stats, test_stats = traintestRetrieve(queries)
	print('TRAIN: ',train_stats,'\nTEST: ',test_stats,'\n')

