from sklearn.neighbors import NearestNeighbors
from util import generatedDataLoading as loading
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from sklearn import metrics


def find_k_nearest_neighbors(data, k):
	assert data.shape[0] == data.shape[1]
	res = np.zeros((data.shape[0], k))
	for i in range(data.shape[0]):
		min_distance_index_set = []
		min_distance_set = []
		for j in range(data.shape[1]):
			if i == j:
				continue
			else:
				if  len(min_distance_set) < k or data[i,j] < max(min_distance_set):
					min_distance_set += [data[i,j]]
					min_distance_index_set += [j]
		res[i] = np.array(min_distance_index_set)[np.array(min_distance_set).argsort()]
	return res

def cal_auc(gt, anomaly_score):
	fpr, tpr, thresholds = metrics.roc_curve(gt, anomaly_score)
	auc = metrics.auc(fpr, tpr)
	return auc

if __name__ == "__main__":
	data, groundTruth = loading.get_data("cardio")
	# data = data[:5]
	# print(data.shape)
	eu_distance = euclidean_distances(data, data)
	# print(eu_distance.shape)
	# # print(eu_distance)
	# neigbors = find_k_nearest_neighbors(eu_distance, 5)
	# print(neigbors)


	# product = np.dot(data, data.T)
	# sorted_index = np.argsort(product)
	
	# eu_distance = np.dot(data, data.T)
	sorted_index = np.argsort(eu_distance)

	for k in [5,25,50,100,200]:
		index = []
		anomaly_score = []
		for i in range(data.shape[0]):
			# print(np.where(sorted_index[i] == 200)[0][0])
			# exit(0)
			index.append(np.where(sorted_index[i] == k)[0][0])
			# anomaly_score.append(product[i, index[i]])
			anomaly_score.append(eu_distance[i, index[i]])
			
		# print(anomaly_score)
		print(cal_auc(groundTruth, anomaly_score))