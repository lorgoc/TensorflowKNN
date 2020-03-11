import numpy as np
import os

dataset_list = ['cardio', 'ecoli', 'kddcup99', 'lymphography', 'waveform', 'waveform_noise', 'kddcup99_sampled']
data_path = "H:/data/" if os.path.isdir("H:/") else "/Users/lappe-rutgers/data/"

def get_data(dataset_name):
	assert dataset_name in dataset_list
	return np.load(data_path+str(dataset_name)+'/data.npy'), np.load(data_path+str(dataset_name)+'/gt.npy')

if __name__ == '__main__':
	for name in dataset_list:
		data,gt = get_data(name)
		print(data.shape)
		print(gt.shape)
