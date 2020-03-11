import numpy as np
import os
from util import vgg16

dataset_list = ['mnist', 'fashion_mnist', 'USPS', 'STL10', '20newsgroups', 'reuters2000']
## TODO reuters
anomaly_rate = 0.1

## dataset_size might be bigger than a single label data volume in set
def get_data(dataset_name, normal):
	assert dataset_name in dataset_list
	if dataset_name == 'mnist':
		return load_mnist(0, normal, anomaly_rate)
	elif dataset_name == 'fashion_mnist':
		return load_fashion_mnist(0, 0, anomaly_rate)
	elif dataset_name == 'USPS':
		return load_USPS(0, 0, anomaly_rate)
	elif dataset_name == 'STL10':
		return load_STL10(3000, 0, anomaly_rate)
	elif dataset_name == '20newsgroups':
		return load_20newsgroups(0, 1, anomaly_rate)
	elif dataset_name == 'reuters2000':		
		# print("Dataset not available.")
		# exit(0)
		return load_reuters2000(2000, normal, anomaly_rate)
	else:
		print("Dataset name not available.")
		exit(0)

def load_mnist(dataset_size, normal, anomaly_rate=0.1):
	from tensorflow.keras.datasets import mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x = np.concatenate((x_train, x_test), axis=0)
	x = x.reshape((x.shape[0],-1))
	y = np.concatenate((y_train, y_test), axis=0)
	z = {'data': x, 'label': y}
	return make_anomaly_detection_dataset(z=z, dataset_size=dataset_size, normal=normal, anomaly_rate=anomaly_rate)

def load_fashion_mnist(dataset_size, normal, anomaly_rate=0.1):
	from tensorflow.keras.datasets import fashion_mnist
	(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
	x = np.concatenate((x_train, x_test), axis=0)
	x = x.reshape((x.shape[0],-1))
	y = np.concatenate((y_train, y_test), axis=0)
	z = {'data': x, 'label': y}
	return make_anomaly_detection_dataset(z=z, dataset_size=dataset_size, normal=normal, anomaly_rate=anomaly_rate)

def load_USPS(dataset_size, normal, anomaly_rate=0.1):
	if os.path.isdir("H:/"):
		data_path = "H:/data/USPS/"
	else:
		data_path = "/home/LAB/liusz/data/USPS/"
	z = []
	for file in ['./zip.test', '/zip.train']:
		with open(data_path + file, 'r') as f:
			lines = f.readlines()
			for line in lines:
				one_line = np.fromstring(line, dtype=np.float32, sep=' ')
				z.append(one_line)
	z = np.array(z)
	x = z[:,1:]
	y = z[:,0]
	z = {'data': x, 'label': y}
	return make_anomaly_detection_dataset(z=z, dataset_size=dataset_size, normal=normal, anomaly_rate=anomaly_rate)

def load_cifar10(dataset_size, normal, anomaly_rate=0.1):
	if os.path.isdir("H:/"):
		data_path = "H:/data/cifar10/"
	else:
		data_path = "/home/LAB/liusz/data/cifar10/"
	# from tensorflow.keras.datasets import cifar10
	# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
	# x = np.concatenate((x_train, x_test), axis=0)
	# x = vgg16.extract_vgg16_features(x)
	# y = np.concatenate((y_train, y_test), axis=0)
	x = np.load('cifar10-vggfeatures-'+str(normal))
	y = np.load('cifar10-groundTruth-'+str(normal))
	z = {'data': x, 'label': y}
	return make_anomaly_detection_dataset(z=z, dataset_size=dataset_size, normal=normal, anomaly_rate=anomaly_rate)


### 'cifar10-vggfeatures-'+str(i)
def load_STL10(dataset_size, normal, anomaly_rate=0.1):
	if os.path.isdir("H:/"):
		data_path = "H:/data/STL10/"
	else:
		data_path = "/home/LAB/liusz/data/STL10/"
	# x1 = np.fromfile(data_path + '/train_X.bin', dtype=np.uint8)
	# x1 = x1.reshape((int(x1.size/3/96/96), 3, 96, 96)).transpose((0, 3, 2, 1))
	# x2 = np.fromfile(data_path + '/test_X.bin', dtype=np.uint8)
	# x2 = x2.reshape((int(x2.size/3/96/96), 3, 96, 96)).transpose((0, 3, 2, 1))
	# x = np.concatenate((x1, x2)).astype(float)
	# y1 = np.fromfile(data_path + '/train_y.bin', dtype=np.uint8) - 1
	# y2 = np.fromfile(data_path + '/test_y.bin', dtype=np.uint8) - 1
	# y = np.concatenate((y1, y2))
	# features = vgg16.extract_vgg16_features(x)

	if normal == 0:
		x = np.load(data_path+"STL10-vggfeatures-animals.npy")
		y = np.load(data_path+"STL10-groundTruth-animals.npy")
	elif normal == 1:
		x = np.load(data_path+"STL10-vggfeatures-transportations.npy")
		y = np.load(data_path+"STL10-groundTruth-transportations.npy")
	z = {'data': x, 'label': y}
	return make_anomaly_detection_dataset(z=z, dataset_size=dataset_size, normal=normal, anomaly_rate=anomaly_rate)


def maping_to_category(y):
	return {
		0 : 0,
		1 : 1,
		2 : 1,
		3 : 1,
		4 : 1,
		5 : 1,
		6 : 2,
		7 : 3,
		8 : 3,
		9 : 3,
		10 : 3,
		11 : 4,
		12 : 4,
		13 : 4,
		14 : 4,
		15 : 5,
		16 : 6,
		17 : 6,
		18 : 6,
		19 : 6
	}[y]

### Counter({1: 4891, 3: 3979, 4: 3952, 6: 3253, 5: 997, 2: 975, 0: 799})
### 1,3,4,6
def load_20newsgroups(dataset_size, normal, anomaly_rate=0.1):
	assert normal in [0,1,2,3,4,5,6]
	if os.path.isdir("H:/"):
		data_path = "H:/data/20newsgroups/"
	else:
		data_path = "/home/LAB/liusz/data/20newsgroups/"
	from sklearn.feature_extraction.text import TfidfVectorizer
	from sklearn.datasets import fetch_20newsgroups
	newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
	vectorizer = TfidfVectorizer(max_features=2000, dtype=np.float64, sublinear_tf=True)
	x_sparse = vectorizer.fit_transform(newsgroups.data)
	x = np.asarray(x_sparse.todense())
	y = newsgroups.target
	y = np.array([maping_to_category(yi) for yi in y])
	names_ = newsgroups.target_names
	z = {'data': x, 'label': y}
	return make_anomaly_detection_dataset(z=z, dataset_size=dataset_size, normal=normal, anomaly_rate=anomaly_rate)

# def load_reuters(dataset_size, normal, anomaly_rate):
# 	normal = 1
# 	if os.path.isdir("H:/"):
# 		data_path = "H:/data/reuters/"
# 	else:
# 		data_path = "/home/LAB/liusz/data/reuters"
# 	x = np.load(data_path + "reuters-data.npy")
# 	y = np.load(data_path + "reuters-groundTruth.npy")
# 	z = {'data': x, 'label': y}
# 	return make_anomaly_detection_dataset(z=z, dataset_size=dataset_size, normal=normal, anomaly_rate=anomaly_rate)

def load_reuters800(dataset_size, normal, anomaly_rate):
	if os.path.isdir("H:/"):
		data_path = "H:/data/reuters/"
	else:
		data_path = "/home/LAB/liusz/data/reuters/"
	x = np.load(data_path + "data800.npy")
	y = np.load(data_path + "groundTruth800.npy")
	z = {'data': x, 'label': y}
	return make_anomaly_detection_dataset(z=z, dataset_size=dataset_size, normal=normal, anomaly_rate=anomaly_rate)

def load_reuters2000(dataset_size, normal, anomaly_rate):
	if os.path.isdir("H:/"):
		data_path = "H:/data/reuters/"
	else:
		data_path = "/home/LAB/liusz/data/reuters/"
	x = np.load(data_path + "data.npy")
	y = np.load(data_path + "groundTruth.npy")
	z = {'data': x, 'label': y}
	return make_anomaly_detection_dataset(z=z, dataset_size=dataset_size, normal=normal, anomaly_rate=anomaly_rate)

## dataset_size == 0 means use all negative data.
def make_anomaly_detection_dataset(z, dataset_size, normal, anomaly_rate):
	x = z['data']
	y = z['label']
	from collections import Counter
	print(Counter(y))
	if dataset_size == 0:
		if y[y==normal].shape[0] / (1 - anomaly_rate) <= y[y!=normal].shape[0] / anomaly_rate:
			normal_size = y[y==normal].shape[0]
			anomaly_size = int(normal_size / (1 - anomaly_rate) * anomaly_rate)
			print(1)
		else:
			anomaly_size = y[y!=normal].shape[0]
			normal_size = int(anomaly_size / anomaly_rate * (1 - anomaly_rate))
			print(2)
	else:
		if y[y==normal].shape[0] > dataset_size * (1 - anomaly_rate) and y[y!=normal].shape[0] > dataset_size * anomaly_rate:
			normal_size = int(dataset_size * (1 - anomaly_rate))
			anomaly_size = int(dataset_size * anomaly_rate)
			print(3)
		elif y[y==normal].shape[0] < dataset_size * (1 - anomaly_rate) and y[y!=normal].shape[0] > dataset_size * anomaly_rate:
			normal_size = y[y==normal].shape[0]
			anomaly_size = (normal_size / (1 - anomaly_rate) * anomaly_rate)
			print(4)
		elif y[y==normal].shape[0] > dataset_size * (1 - anomaly_rate) and y[y!=normal].shape[0] < dataset_size * anomaly_rate:
			anomaly_size = y[y!=normal].shape[0]
			normal_size = int(anomaly_size / anomaly_rate * (1 - anomaly_rate))
			print(5)
		else:
			if y[y==normal].shape[0] / (1 - anomaly_rate) <= y[y!=normal].shape[0] / anomaly_rate:
				normal_size = y[y==normal].shape[0]
				anomaly_size = int(normal_size / (1 - anomaly_rate) * anomaly_rate)
				print(6)
			else:
				anomaly_size = y[y!=nomal].shape[0]
				normal_size = int(anomaly_size / anomaly_rate * (1 - anomaly_rate))
				print(7)
	y = np.reshape(y, (-1,1))
	# print(x.shape, y.shape)
	x_y = np.concatenate((x,y), axis=1)
	np.random.shuffle(x_y)
	x = x_y[:,:-1]
	y = x_y[:,-1]
	out_y = np.zeros((normal_size + anomaly_size,))
	out_x = x[y==normal]
	out_x = out_x[:normal_size]
	list_types = list(np.unique(y))
	one_type_anomaly_size = int(anomaly_size / (len(list_types) - 1))
	# for anomaly_type in list_types:
	# 	if anomaly_type == normal:
	# 		continue
	# 	else:
	# 		out_x = np.concatenate((out_x, x[y==anomaly_type][:one_type_anomaly_size]),axis=0)
	out_x = np.concatenate((out_x, x[y!=normal][:anomaly_size]))
	assert out_x.shape[0] == anomaly_size + normal_size
	out_y = np.zeros((out_x.shape[0],))
	out_y[normal_size:] = 1
	out_y = np.reshape(out_y, (-1,1))
	out_z = np.concatenate((out_x,out_y), axis=1)
	np.random.shuffle(out_z)
	print(out_z.shape[0], out_z.shape[1]-1)
	from sklearn.preprocessing import MinMaxScaler
	print("MinMaxScaling")
	return MinMaxScaler().fit_transform(out_z[:,:-1]), out_z[:,-1]



if __name__ == "__main__":
	import time
	t = time.time()
	data, groundTruth = get_data('reuters2000')
	print(time.time() - t)
	print(data.shape)
	from collections import Counter
	print(Counter(groundTruth))
