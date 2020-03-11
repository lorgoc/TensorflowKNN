from keras.preprocessing.image import img_to_array, array_to_img
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.models import Model
import numpy as np

def extract_vgg16_features(x):
	im_h = 224
	model = VGG16(include_top=True, weights='imagenet', input_shape=(im_h, im_h, 3))
	feature_model = Model(model.input, model.get_layer('fc1').output)
	#model = VGG16(weights='imagenet', include_top=False)

	print('extracting VGG features')
	x = np.asarray([img_to_array(array_to_img(im, scale=False).resize((im_h,im_h))) for im in x])
	x = preprocess_input(x)  

	# from keras documents
	#features = model.predict(x)  # output dimensionality is 1000
	features = feature_model.predict(x)  # output dimensionality is 4096
	print('VGG feature shape: ', features.shape)

	return features