'''
Purpose: Loads .jpg images into numpy arrays, tags them and stores image and label in dictionary, sends to .npy file for easy access.
'''

import os
import cv2
import glob
import numpy as np
from sklearn.utils import shuffle

class data_loader():
	def __init__(self, data_filepath=''):
		'''
		Initializes data filepath, the labels for AD types, and empty dicts for the data.
		'''
		if data_filepath == '':
			self.filepath = os.path.abspath(os.path.join(os.getcwd(),'data'))
		else:
			self.filepath == data_filepath

		self.ad_types = ['MildDemented','ModerateDemented','NonDemented','VeryMildDemented']
		self.train_data = {'images':[], 'labels':[]}
		self.test_data = {'images':[], 'labels':[]}
		self.all_data = {'images':[], 'labels':[], 'binary_labels':[]}

	def load_images(self):
		'''
		Loads train and test images, calls _apply_labels() to add labels, adds to train/test dicts.
		'''
		test_images = []
		for label in self.ad_types:
			train_files = glob.glob(self.filepath+'/train/'+label+'/*.jpg')
			test_files = glob.glob(self.filepath+'/test/'+label+'/*.jpg')

			train_images = []
			for file in train_files:
				image = cv2.imread(file)
				#image = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY)
				train_images.append(image)

			tagged_images = self._apply_labels(train_images, label)

			self.train_data['images'] = self.train_data['images'] + tagged_images['images']
			self.train_data['labels'] = self.train_data['labels'] + tagged_images['labels']

			test_images = []
			for file in test_files:
				image = cv2.imread(file)
				#image = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY)
				test_images.append(image)

			tagged_images = self._apply_labels(test_images, label)

			self.test_data['images'] = self.test_data['images'] + tagged_images['images']
			self.test_data['labels'] = self.test_data['labels'] + tagged_images['labels']

	def _apply_labels(self, images, label):
		'''
		Combines the images with the correct label .
		'''
		labels_len = len(images)
		labels = [label] * labels_len

		tagged_images = {'images':images, 'labels':labels}

		return tagged_images

	def merge_data(self):
		'''
		Merges the train and test data into a single dataset for analysis and custom train/test splits.
		Requires load_data() to have been successfully ran.
		'''
		images = self.train_data['images'] + self.test_data['images']
		labels = self.train_data['labels'] + self.test_data['labels']

		# shuffle data while maintaining image/label relationship
		images, labels = shuffle(np.asarray(images), np.asarray(labels), random_state=0)

		self.all_data['images'] = images
		self.all_data['labels'] = labels

	def extract_data(self):
		'''
		Sends merged data to .npy (numpy) files for quicker access when performing analysis and modeling.
		Requires merge_data() to have been successfully ran.
		'''
		np.save(self.filepath+'/image_arrays.npy', self.all_data['images'])
		np.save(self.filepath+'/image_labels.npy', self.all_data['labels'])
		np.save(self.filepath+'/binary_image_labels.npy', self.all_data['binary_labels'])

	def create_binary_labels(self):
		'''
		Creates labels for identifying whether an image has any AD or not.
		'''
		binary_labels = []
		for label in self.all_data['labels']:
			if label != 'NonDemented':
				binary_labels.append(1)
			else:
				binary_labels.append(0)

		self.all_data['binary_labels'] = binary_labels

'''
import data_loader as dl
d = dl.data_loader()
d.load_images()
d.merge_data()
d.create_binary_labels()
d.extract_data()

'''