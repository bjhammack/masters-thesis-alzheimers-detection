'''
Purpose: Creates additional image samples based on analysis performed earlier.

Goal: Oversample ModerateDemented, and MildDemented images to be comparable with VeryMildDemented size and add 
NonDemented samples to be equal to the new count of Demented samples.
'''
import numpy as np
import random as ran

ran.seed(12)

class Oversampler():
	def __init__(self, x='', y='', oversample_ratio=1.0):
		if len(x) < 1 or len(y) < 1:
			raise Exception('Data or labels not found. Cannot perform oversampling.')
		elif str(type(x)) != "<class 'numpy.ndarray'>":
			raise Exception('Data needs to be a numpy array.')
		else:
			self.x = x
			self.y = y
			self.oversample_ratio = oversample_ratio
			self.distances = []

	def smote(self, n_neighbors=5):
		artificial_images = {'MildDemented':[],'ModerateDemented':[],'VeryMildDemented':[]}
		new_image_count = {'MildDemented':0,'ModerateDemented':0,'VeryMildDemented':0}
		
		used_sets = []

		image_sets, oversample_count = self.get_oversample_count()

		for k,v in oversample_count.items():
			if oversample_count[k] > 0:
				for i in range(0,int(oversample_count[k])):
					image = ran.choice(image_sets[k])

					neighbors = self._get_knn(image, image_sets[k].copy(), n_neighbors)
					is_dupe = self._is_duplicate(used_sets, neighbors, n_neighbors)

					if is_dupe == False:
						artificial_image = np.mean(neighbors, axis=0)
						artificial_images[k].append(artificial_image)

						used_sets.append(neighbors)
						new_image_count[k] += 1
			else:
				pass

		return artificial_images, new_image_count

	def _euclidean_distance(self, image1, image2):
		diff_sqrd = np.sum((image1 - image2)**2)
		distance = np.sqrt(diff_sqrd)

		return distance

	def get_oversample_count(self):
		sample_size = len(self.x)
		images_dict = {'MildDemented':[],'ModerateDemented':[],'VeryMildDemented':[]}

		for i in range(0,sample_size):
			for k,v in images_dict.items():
				if self.y[i] == k:
					images_dict[k].append(self.x[i])

		largest = 0
		artificial_image_count = {'MildDemented':0,'ModerateDemented':0,'VeryMildDemented':0}
		for k,v in images_dict.items():
			if len(v) > largest:
				largest = len(v)

		for k,v in artificial_image_count.items():
			artificial_image_count[k] = int((largest - len(images_dict[k])) * self.oversample_ratio)

		return images_dict, artificial_image_count

	def _get_knn(self, image, dataset, n_neighbors=5):
		neighbors = []
		distances = []

		dataset = self._remove_array_element(dataset,image)

		for i in range(0,len(dataset)):
			distance = self._euclidean_distance(image, dataset[i])
			distances.append(distance)

		for i in range(0,n_neighbors):
			min_index = distances.index(min(distances))
			#if distances[min_index] < 2200:
			neighbors.append(dataset[min_index])
			self.distances.append(distances[min_index])
			distances.pop(min_index)

		return neighbors

	def _remove_array_element(self, array_list, target_array):
		index = 0
		size = len(array_list)
		while index != size and not np.array_equal(array_list[index],target_array):
			index += 1

		if index != size:
			array_list.pop(index)
		else:
			raise ValueError('Cannot find array to remove from list.')

		return array_list

	def _is_duplicate(self, used_sets, neighbors, n_neighbors):
		for used_set in used_sets:
			dupe_count = 0
			for neighbor in neighbors:
				match = next((True for elem in used_set if np.array_equal(elem, neighbor)), False)
				if match:
					dupe_count += 1
			if dupe_count == n_neighbors:
				return True
		return False


'''
import oversampler as ov
o = ov.oversampler(data['images'],data['labels'])
o.smote()

'''